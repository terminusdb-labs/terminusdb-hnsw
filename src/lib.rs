#![no_std]
extern crate alloc;

mod hnsw;

pub use self::hnsw::*;

use ahash::RandomState;
use alloc::{vec, vec::Vec};
use convenient_skiplist::SkipList;
use hashbrown::HashSet;
use space::Neighbor;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Params {
    ef_construction: usize,
}

impl Params {
    pub fn new() -> Self {
        Default::default()
    }

    /// This is refered to as `efConstruction` in the paper. This is equivalent to the `ef` parameter passed
    /// to `nearest`, but it is the `ef` used when inserting elements. The higher this is, the more likely the
    /// nearest neighbors in each graph level will be correct, leading to a higher recall rate and speed when
    /// calling `nearest`. This parameter greatly affects the speed of insertion into the HNSW.
    ///
    /// This parameter is probably the only one that in important to tweak.
    ///
    /// Defaults to `400` (overkill for most tasks, but only lower after profiling).
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            ef_construction: 400,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NeighborForHeap<Unit: PartialEq + Eq + PartialOrd + Ord>(pub Neighbor<Unit>);

impl<Unit: PartialEq + Eq + PartialOrd + Ord> PartialOrd for NeighborForHeap<Unit> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Unit: PartialEq + Eq + PartialOrd + Ord> Ord for NeighborForHeap<Unit> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.distance.cmp(&other.0.distance)
    }
}

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug)]
pub struct Searcher<Metric: Ord> {
    candidates: Vec<Neighbor<Metric>>,
    nearest: SkipList<NeighborForHeap<Metric>>,
    seen: HashSet<usize, RandomState>,
}

impl<Metric: Ord + Clone> Searcher<Metric> {
    pub fn new() -> Self {
        Default::default()
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}

impl<Metric: Ord + Clone> Default for Searcher<Metric> {
    fn default() -> Self {
        Self {
            candidates: vec![],
            nearest: SkipList::new(),
            seen: HashSet::with_hasher(RandomState::with_seeds(0, 0, 0, 0)),
        }
    }
}
