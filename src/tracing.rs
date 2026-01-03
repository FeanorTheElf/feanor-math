use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::{Display, Write};
use std::io::stdout;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::RangeInclusive;
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

use thread_local::ThreadLocal;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Metadata, Subscriber, span};
use tracing::span::Id;
use tracing_core::Interest;

struct LoggingPermission {
    data: PhantomData<()>
}

impl LoggingPermission {

    fn create_new_permission() -> Self {
        LoggingPermission { data: PhantomData }
    }
}

struct LoggingPermissionHolder {
    data: AtomicBool
}

impl LoggingPermissionHolder {

    fn new() -> Self {
        Self { data: AtomicBool::new(false) }
    }

    fn take(&self) -> Option<LoggingPermission> {
        if self.data.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            Some(LoggingPermission::create_new_permission())
        } else {
            None
        }
    }

    fn has_permission(&self) -> bool {
        self.data.load(Ordering::SeqCst)
    }

    fn give(&self, _permission: LoggingPermission) {
        assert!(self.data.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok())
    }
}

struct SpanState {
    parent: Option<Id>,
    /// how many handles to the span currently exist; used for reference counting,
    /// to remove the span from map of spans when it is no longer used
    reference_counter: AtomicUsize,
    /// custom string with additional info about the span
    description: String,
    /// span metadata
    metadata: &'static Metadata<'static>,
    /// whether the span is currently allowed to produce logs; A permission is created for
    /// every root span, and moved (not copied) to the first entered child span. Once a span
    /// is exited, the permission is given back to the parent span.
    logging_permission: LoggingPermissionHolder,
    /// Messages that would be printed during the first `span_min_duration` microseconds of the
    /// span are instead queued here, and printed with the first print that happens after
    /// `span_min_duration` are passed
    queued_messages: Mutex<Vec<String>>,
    /// whether the queued messages have been printed
    printed_queued_messages: AtomicBool,
    /// dual purpose: track time, and ensure that a span is not entered by multiple threads
    /// at the same time; a span that is not entered has the value `0` here
    entered_timestamp: AtomicU64,
    /// on which level of the span tree this span is
    level: usize,
}

pub struct LogAlgorithmSubscriber {
    span_ids: AtomicU64,
    span_map: RwLock<HashMap<Id, SpanState>>,
    current_span: ThreadLocal<Cell<Option<NonZeroU64>>>,
    default_instant: Instant,
    interested_level: RangeInclusive<Level>,
    max_depth: usize,
    span_min_duration: u64
}

impl LogAlgorithmSubscriber {

    ///
    /// Initializes a [`LogAlgorithmsSubscriber`], which will print to stdout all
    /// tracing `span!`s and `events!` that match the following:
    ///  - the tracing level is within `levels`
    ///  - the spans are nested at most `max_depth` deep
    ///  - the current span is running for at least `span_min_duration_in_micros` microseconds
    /// 
    pub fn init(levels: RangeInclusive<Level>, max_depth: usize, span_min_duration_in_micros: u64) {
        tracing::subscriber::set_global_default(Self { 
            span_ids: AtomicU64::new(1), 
            span_map: RwLock::new(HashMap::new()), 
            current_span: ThreadLocal::new(),
            default_instant: Instant::now(),
            max_depth: max_depth,
            interested_level: levels,
            span_min_duration: span_min_duration_in_micros
        }).unwrap()
    }

    pub fn init_test() {
        _ = tracing::subscriber::set_global_default(Self { 
            span_ids: AtomicU64::new(1), 
            span_map: RwLock::new(HashMap::new()), 
            current_span: ThreadLocal::new(),
            default_instant: Instant::now(),
            interested_level: Level::ERROR..=Level::TRACE,
            max_depth: 4,
            span_min_duration: 100000
        })
    }

    fn span_map<'a>(&'a self) -> RwLockReadGuard<'a, HashMap<Id, SpanState>> {
        self.span_map.read().unwrap()
    }
    
    fn span_map_mut<'a>(&'a self) -> RwLockWriteGuard<'a, HashMap<Id, SpanState>> {
        self.span_map.write().unwrap()
    }

    fn take_permission_for_span(&self, span_map: &RwLockReadGuard<HashMap<Id, SpanState>>, span: &SpanState) {
        if span.logging_permission.has_permission() {
            if let Ok(_) = span.printed_queued_messages.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst) {
                for message in span.queued_messages.lock().unwrap().drain(..) {
                    print!("{}", message);
                }
                std::io::Write::flush(&mut stdout()).unwrap();
            }
            return;
        }
        if let Some(parent) = &span.parent {
            let parent = span_map.get(parent).unwrap();
            self.take_permission_for_span(span_map, parent);
            if let Some(permission) = parent.logging_permission.take() {
                // println!("taking permission from parent {} to {}", parent.description, span.description);
                if let Ok(_) = span.printed_queued_messages.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst) {
                    for message in span.queued_messages.lock().unwrap().drain(..) {
                        print!("{}", message);
                    }
                    std::io::Write::flush(&mut stdout()).unwrap();
                }
                span.logging_permission.give(permission);
            }
        }
    }

    ///
    /// If the current span has been running for at least `span_min_duration` microseconds,
    /// this function will try to obtain the logging permission, and if granted, log the given
    /// message. Otherwise, they will be queued and printed together with the first message
    /// at which the span has been running for at least that long, assuming there is any.
    /// 
    /// Note that in cases that the time given by `Instant::now()` is non-monotonous, rare
    /// cases of wrong (dropped or misordered) logging may occur.
    /// 
    fn delayed_print_messages<'a, I: IntoIterator<Item = String>>(&self, current_span: Option<&SpanState>, span_map: &RwLockReadGuard<HashMap<Id, SpanState>>, messages: I) {
        if let Some(current_span) = current_span {
            if current_span.printed_queued_messages.load(Ordering::SeqCst) {
                // println!("directly printing");
                if current_span.logging_permission.has_permission() {
                    for message in messages {
                        print!("{}", message);
                    }
                    std::io::Write::flush(&mut stdout()).unwrap();
                }
                return;
            }
            let runtime = self.default_instant.elapsed().as_micros() as u64 - current_span.entered_timestamp.load(Ordering::SeqCst);
            if runtime > self.span_min_duration {
                self.take_permission_for_span(span_map, current_span);
                if current_span.logging_permission.has_permission() {
                    for message in messages {
                        print!("{}", message);
                    }
                    std::io::Write::flush(&mut stdout()).unwrap();
                }
            } else {
                let messages = messages.into_iter().collect::<Vec<_>>();
                // println!("queueing {} -> {}", &messages[0], current_span.description);
                current_span.queued_messages.lock().unwrap().extend(messages);
            }
        } else {
            for message in messages {
                print!("{}", message);
            }
            std::io::Write::flush(&mut stdout()).unwrap();
        }
    }
}

struct FieldRecorder {
    message: Option<String>,
    fields: Option<String>
}

impl FieldRecorder {

    fn new() -> Self {
        Self { message: None, fields: None }
    }

    fn to_string(self) -> String {
        match (self.message, self.fields) {
            (Some(mut message), Some(fields)) => {
                write!(&mut message, "({})", fields).unwrap();
                message
            },
            (Some(mut message), None) => {
                write!(&mut message, ".").unwrap();
                message
            },
            (None, Some(fields)) => {
                format!("({})", fields)
            },
            (None, None) => ".".to_owned()
        }
    }
}

impl Display for FieldRecorder {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(message) = &self.message {
            write!(f, "{}", message)?;
        }
        if let Some(fields) = &self.fields {
            write!(f, "({})", fields)
        } else {
            write!(f, ".")
        }
    }
}

impl Visit for FieldRecorder {

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{:?}", value));
        } else {
            if let Some(fields) = &mut self.fields {
                write!(fields, ", {}={:?}", field.name(), value).unwrap();
            } else {
                self.fields = Some(format!("{}={:?}", field.name(), value));
            }
        }
    }
}

impl Subscriber for LogAlgorithmSubscriber {

    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if self.interested_level.contains(metadata.level()) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.interested_level.contains(metadata.level())
    }

    fn current_span(&self) -> tracing_core::span::Current {
        if let Some(span) = self.current_span.get_or(|| Cell::new(None)).get() {
            tracing_core::span::Current::new(Id::from_non_zero_u64(span), self.span_map().get(&Id::from_non_zero_u64(span)).unwrap().metadata)
        } else {
            tracing_core::span::Current::none()
        }
    }

    fn new_span(&self, span: &span::Attributes<'_>) -> Id {
        let id = NonZeroU64::try_from(self.span_ids.fetch_add(1, Ordering::Relaxed)).unwrap();
        let mut spans = self.span_map_mut();
        let parent = span.parent().cloned().or_else(|| self.current_span.get_or(|| Cell::new(None)).get().map(Id::from_non_zero_u64));
        let level = parent.as_ref().map(|id| spans.get(id).unwrap().level + 1).unwrap_or(0);

        let mut fields = FieldRecorder::new();
        span.record(&mut fields);
        fields.message = Some(span.metadata().name().to_owned());

        let logging_permission = LoggingPermissionHolder::new();
        if parent.is_none() {
            logging_permission.give(LoggingPermission::create_new_permission());
        }

        assert!(spans.insert(Id::from_non_zero_u64(id), SpanState {
            parent: parent,
            level: level,
            metadata: span.metadata(),
            reference_counter: AtomicUsize::new(1),
            description: fields.to_string(),
            logging_permission: logging_permission,
            entered_timestamp: AtomicU64::new(0),
            printed_queued_messages: AtomicBool::new(false),
            queued_messages: Mutex::new(Vec::new())
        }).is_none());
        return Id::from_non_zero_u64(id);
    }

    fn record(&self, _span: &Id, _values: &span::Record<'_>) {
        unimplemented!()
    }

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {
        // we only care about parent spans currently
    }

    fn event(&self, event: &Event<'_>) {
        if self.interested_level.contains(event.metadata().level()) {
            let span_map = self.span_map();
            let current_span = self.current_span().id().map(|id| span_map.get(id).unwrap());
            if current_span.is_none() || (current_span.unwrap().logging_permission.has_permission() && current_span.unwrap().level < self.max_depth) {
                let mut fields = FieldRecorder::new();
                event.record(&mut fields);
                self.delayed_print_messages(current_span, &span_map, [fields.to_string()]);
            }
        }
    }

    fn enter(&self, span: &Id) {
        self.current_span.get_or(|| Cell::new(None)).set(Some(span.into_non_zero_u64()));
        let span_map = self.span_map();
        let entered_span = span_map.get(span).unwrap();
        assert!(entered_span.entered_timestamp.compare_exchange(0, Instant::now().duration_since(self.default_instant).as_micros() as u64, Ordering::SeqCst, Ordering::SeqCst).is_ok(), "entered an already running span");
        if entered_span.level < self.max_depth {
            self.delayed_print_messages(Some(entered_span), &span_map, [entered_span.description.clone()]);
        } else if entered_span.level == self.max_depth {
            self.delayed_print_messages(Some(entered_span), &span_map, [format!("{}...", entered_span.description)]);
        }
    }

    fn exit(&self, span: &Id) {
        let span_map = self.span_map();
        let exited_span = span_map.get(span).unwrap();
        let time = Instant::now().duration_since(self.default_instant).as_micros() as u64 - exited_span.entered_timestamp.load(Ordering::SeqCst);
        if exited_span.level <= self.max_depth {
            if exited_span.level == 0 {
                self.delayed_print_messages(Some(exited_span), &span_map, [format!("done({}us)\n", time)]);
            } else { 
                self.delayed_print_messages(Some(exited_span), &span_map, [format!("done({}us)", time)]);
            }
        }
        if let Some(logging_permission) = exited_span.logging_permission.take() {
            if let Some(parent) = &exited_span.parent {
                span_map.get(&parent).unwrap().logging_permission.give(logging_permission);
            }
        }
        self.current_span.get_or(|| Cell::new(None)).set(exited_span.parent.as_ref().map(|id| id.into_non_zero_u64()));
        exited_span.entered_timestamp.store(0, Ordering::SeqCst);
    }

    fn clone_span(&self, id: &Id) -> Id {
        _ = self.span_map().get(id).unwrap().reference_counter.fetch_add(1, Ordering::Relaxed);
        return id.clone();
    }

    fn try_close(&self, id: Id) -> bool {
        let remaining_handles = self.span_map().get(&id).unwrap().reference_counter.fetch_sub(1, Ordering::Relaxed) - 1;
        if remaining_handles == 0 {
            _ = self.span_map_mut().remove(&id).unwrap();
            true
        } else {
            false
        }
    }
}
