use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::{Display, Write};
use std::io::stdout;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::RangeInclusive;
use std::sync::RwLock;
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
    /// whether the span is currently allowed to produce logs
    logging_permission: LoggingPermissionHolder,
    /// to ensure that a span is not entered by multiple threads at the same time
    entered_timestamp: AtomicU64,
    /// on which level of the span tree this span is
    level: usize
}

pub struct LogAlgorithmSubscriber {
    span_ids: AtomicU64,
    span_map: RwLock<HashMap<Id, SpanState>>,
    current_span: ThreadLocal<Cell<Option<NonZeroU64>>>,
    default_instant: Instant,
    interested_level: RangeInclusive<Level>,
    max_depth: usize
}

impl LogAlgorithmSubscriber {

    pub fn init(levels: RangeInclusive<Level>, max_depth: usize) {
        tracing::subscriber::set_global_default(Self { 
            span_ids: AtomicU64::new(1), 
            span_map: RwLock::new(HashMap::new()), 
            current_span: ThreadLocal::new(),
            default_instant: Instant::now(),
            max_depth: max_depth,
            interested_level: levels
        }).unwrap()
    }

    pub fn init_test() {
        _ = tracing::subscriber::set_global_default(Self { 
            span_ids: AtomicU64::new(1), 
            span_map: RwLock::new(HashMap::new()), 
            current_span: ThreadLocal::new(),
            default_instant: Instant::now(),
            interested_level: Level::INFO..=Level::INFO,
            max_depth: 2
        })
    }

    fn span_map<'a>(&'a self) -> RwLockReadGuard<'a, HashMap<Id, SpanState>> {
        self.span_map.read().unwrap()
    }
    
    fn span_map_mut<'a>(&'a self) -> RwLockWriteGuard<'a, HashMap<Id, SpanState>> {
        self.span_map.write().unwrap()
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

        let mut description = FieldRecorder::new();
        span.record(&mut description);
        description.message = Some(span.metadata().name().to_owned());

        assert!(spans.insert(Id::from_non_zero_u64(id), SpanState {
            parent: parent,
            level: level,
            metadata: span.metadata(),
            reference_counter: AtomicUsize::new(1),
            description: description.to_string(),
            logging_permission: LoggingPermissionHolder::new(),
            entered_timestamp: AtomicU64::new(0)
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
        if *event.metadata().level() == Level::INFO {
            let span_map = self.span_map();
            let current_span = self.current_span().id().map(|id| span_map.get(id).unwrap());
            if current_span.is_none() || (current_span.unwrap().logging_permission.has_permission() && current_span.unwrap().level < self.max_depth) {
                let mut description = FieldRecorder::new();
                event.record(&mut description);
                print!("{}", description.to_string());
                std::io::Write::flush(&mut stdout()).unwrap();
            }
        }
    }

    fn enter(&self, span: &Id) {
        self.current_span.get_or(|| Cell::new(None)).set(Some(span.into_non_zero_u64()));
        let span_map = self.span_map();
        let entered_span = span_map.get(span).unwrap();
        let permission = if let Some(parent) = &entered_span.parent {
            span_map.get(&parent).unwrap().logging_permission.take()
        } else {
            Some(LoggingPermission::create_new_permission())
        };
        if permission.is_some() {
            if entered_span.level < self.max_depth {
                print!("{}", entered_span.description);
                std::io::Write::flush(&mut stdout()).unwrap();
            }
            if entered_span.level == self.max_depth {
                print!("{}...", entered_span.description);
                std::io::Write::flush(&mut stdout()).unwrap();
            }
        }
        assert!(entered_span.entered_timestamp.compare_exchange(0, Instant::now().duration_since(self.default_instant).as_micros() as u64, Ordering::SeqCst, Ordering::SeqCst).is_ok(), "entered an already running span");
        if let Some(permission) = permission {
            entered_span.logging_permission.give(permission);
        }
    }

    fn exit(&self, span: &Id) {
        let span_map = self.span_map();
        let exited_span = span_map.get(span).unwrap();
        let entered_timestamp = exited_span.entered_timestamp.swap(0, Ordering::SeqCst);
        let time = Instant::now().duration_since(self.default_instant).as_micros() as u64 - entered_timestamp;
        let logging_permission = exited_span.logging_permission.take();
        if let Some(permission) = logging_permission {
            if exited_span.level <= self.max_depth {
                print!("done({}us)", time);
                if exited_span.level == 0 {
                    println!();
                }
                std::io::Write::flush(&mut stdout()).unwrap();
            }
            if let Some(parent) = &exited_span.parent {
                span_map.get(&parent).unwrap().logging_permission.give(permission);
            }
        }
        self.current_span.get_or(|| Cell::new(None)).set(exited_span.parent.as_ref().map(|id| id.into_non_zero_u64()));
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
