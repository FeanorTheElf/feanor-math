use crate::{ring::*};
use crate::algorithms;
use std::cmp::Ordering::*;

#[derive(Clone, Debug)]
pub struct DefaultBigInt(bool, Vec<u64>);

pub struct DefaultBigIntRing;

impl RingBase for DefaultBigIntRing {
    
    type Element = DefaultBigInt;

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        match (lhs, rhs) {
            (DefaultBigInt(false, lhs_val), DefaultBigInt(false, rhs_val)) |
            (DefaultBigInt(true, lhs_val), DefaultBigInt(true, rhs_val)) => {
                algorithms::bigint::bigint_add(lhs_val, rhs_val, 0);
            },
            (DefaultBigInt(lhs_sgn, lhs_val), DefaultBigInt(_, rhs_val)) => {
                match algorithms::bigint::bigint_cmp(lhs_val, rhs_val) {
                    Less => {
                        algorithms::bigint::bigint_sub_self(lhs_val, rhs_val);
                        *lhs_sgn = !*lhs_sgn;
                    },
                    Equal => {
                        lhs_val.clear();
                    },
                    Greater => {
                        algorithms::bigint::bigint_sub(lhs_val, rhs_val, 0);
                    }
                }
            }
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { 
        self.negate_inplace(lhs);
        self.add_assign_ref(lhs, rhs);
        self.negate_inplace(lhs);
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        lhs.0 = !lhs.0;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        let result = algorithms::bigint::bigint_mul(&lhs.1, &rhs.1, Vec::new());
        *lhs = DefaultBigInt(lhs.0 ^ rhs.0, result);
    }

    fn zero(&self) -> Self::Element {
        DefaultBigInt(false, Vec::new())
    }

    fn one(&self) -> Self::Element {
        DefaultBigInt(false, vec![1])
    }

    fn neg_one(&self) -> Self::Element {
        self.negate(self.one())
    }

    fn from_z(&self, value: i32) -> Self::Element {
        DefaultBigInt(value < 0, vec![(value as i64).abs() as u64])
    }

    fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        if lhs.0 == rhs.0 {
            algorithms::bigint::bigint_cmp(&lhs.1, &rhs.1) == Equal
        } else {
            self.is_zero(lhs) && self.is_zero(rhs)
        }
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        algorithms::bigint::highest_set_block(&value.1).is_none()
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        value.0 == false && algorithms::bigint::highest_set_block(&value.1) == Some(0) && value.1[0] == 1
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        value.0 == true && algorithms::bigint::highest_set_block(&value.1) == Some(0) && value.1[0] == 1
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
}