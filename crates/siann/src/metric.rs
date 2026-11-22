//! The metrics: how a distance is built one coordinate at a time, as types the kernels are generic over,
//! and the metric chosen at run time.

use super::Result;

/// How a metric builds a distance one coordinate at a time. L2 distances are squared.
pub trait Metric {
    /// Whether the integer path for byte lists applies: true for the distances; not for the inner
    /// product, whose values are negative.
    const INTEGER: bool = true;
    /// The running distance `acc` with the coordinates `a` and `b` added.
    fn combine(acc: f32, a: f32, b: f32) -> f32;
    /// Two running distances over disjoint coordinates, joined.
    fn join(x: f32, y: f32) -> f32;
    /// The distance between two byte vectors, in integers: exact. A plain loop over each vector's
    /// contiguous bytes, which the compiler may reorder (integer sums are associative), and so
    /// vectorizes 16 bytes at a time (`uabd`, then `udot` for L2).
    fn dist_int(q: &[u8], x: &[u8]) -> u32;
}

/// The sum of |a - b|.
pub struct L1;
/// The sum of (a - b)^2.
pub struct L2;
/// The largest |a - b|.
pub struct Linf;
/// The inner product, negated (the sum of -a b), so that, as for the distances, smaller is nearer.
pub struct Ip;

impl Metric for L1 {
    #[inline(always)]
    fn combine(acc: f32, a: f32, b: f32) -> f32 {
        acc + (a - b).abs()
    }
    #[inline(always)]
    fn join(x: f32, y: f32) -> f32 {
        x + y
    }
    #[inline(always)]
    fn dist_int(q: &[u8], x: &[u8]) -> u32 {
        q.iter()
            .zip(x)
            .map(|(&a, &b)| u32::from(a.abs_diff(b)))
            .sum()
    }
}

impl Metric for L2 {
    #[inline(always)]
    fn combine(acc: f32, a: f32, b: f32) -> f32 {
        let e = a - b;
        acc + e * e
    }
    #[inline(always)]
    fn join(x: f32, y: f32) -> f32 {
        x + y
    }
    #[inline(always)]
    fn dist_int(q: &[u8], x: &[u8]) -> u32 {
        q.iter()
            .zip(x)
            .map(|(&a, &b)| {
                let e = u32::from(a.abs_diff(b));
                e * e
            })
            .sum()
    }
}

// A comparison and select rather than `f32::max`: LLVM treats `max` as a reduction it may reorder,
// and then vectorizes the wrong loop (over dimensions, with shuffles and a 3.7 KB stack frame),
// taking twice L1's time; so does an unsigned-integer max on the bits. The select keeps one running
// maximum per lane, like L1's sums (fcmgt and bit where L1 has one fadd: 1.35 times L1's time on
// 200,000 Gaussian vectors, against 2.1 with `max`). The inputs hold no NaN, so the two agree.
#[inline(always)]
pub fn larger(x: f32, y: f32) -> f32 {
    if y > x { y } else { x }
}

impl Metric for Linf {
    #[inline(always)]
    fn combine(acc: f32, a: f32, b: f32) -> f32 {
        larger(acc, (a - b).abs())
    }
    #[inline(always)]
    fn join(x: f32, y: f32) -> f32 {
        larger(x, y)
    }
    #[inline(always)]
    fn dist_int(q: &[u8], x: &[u8]) -> u32 {
        u32::from(
            q.iter()
                .zip(x)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap_or(0),
        )
    }
}

impl Metric for Ip {
    const INTEGER: bool = false;
    #[inline(always)]
    fn combine(acc: f32, a: f32, b: f32) -> f32 {
        acc - a * b
    }
    #[inline(always)]
    fn join(x: f32, y: f32) -> f32 {
        x + y
    }
    fn dist_int(_q: &[u8], _x: &[u8]) -> u32 {
        unreachable!("the integer path is not used for the inner product")
    }
}

/// The metric an index was built with, chosen at run time; each search dispatches on it once.
#[derive(Clone, Copy)]
pub enum Kind {
    L1,
    L2,
    Linf,
    Ip,
}

impl Kind {
    /// "l1", "l2", "linf" or "ip".
    pub fn parse(metric: &str) -> Result<Kind> {
        match metric {
            "l1" => Ok(Kind::L1),
            "l2" => Ok(Kind::L2),
            "linf" => Ok(Kind::Linf),
            "ip" => Ok(Kind::Ip),
            _ => Err("metric must be 'l1', 'l2', 'linf' or 'ip' (inner product)".into()),
        }
    }

    /// Whether larger scores are better: the inner product's.
    pub fn similarity(self) -> bool {
        matches!(self, Kind::Ip)
    }
}

/// Runs `$body` with `$m` the metric type for `$kind`.
macro_rules! with_metric {
    ($kind:expr, $m:ident => $body:expr) => {
        match $kind {
            $crate::metric::Kind::L1 => {
                type $m = $crate::metric::L1;
                $body
            }
            $crate::metric::Kind::L2 => {
                type $m = $crate::metric::L2;
                $body
            }
            $crate::metric::Kind::Linf => {
                type $m = $crate::metric::Linf;
                $body
            }
            $crate::metric::Kind::Ip => {
                type $m = $crate::metric::Ip;
                $body
            }
        }
    };
}
pub(crate) use with_metric;

/// The score of vector `x` for query `q`: the Euclidean distance for L2, the sums of absolute differences for
/// L1, the largest for L-infinity, the inner product for "ip".
pub fn score(kind: Kind, q: &[f32], x: &[f32]) -> f32 {
    let pairs = q.iter().zip(x);
    match kind {
        Kind::L2 => pairs.map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt(),
        Kind::L1 => pairs.map(|(a, b)| (a - b).abs()).sum(),
        Kind::Linf => pairs.fold(0.0f32, |m, (a, b)| larger(m, (a - b).abs())),
        Kind::Ip => pairs.map(|(a, b)| a * b).sum(),
    }
}
