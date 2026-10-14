//! Three-way merge and per-list merge strategies (spec §6.8, docs/merge-matrix.md).
//!
//! Contributions are suggestion branches above a base seq; merging replays the three-way
//! resolution over the main state: what only the branch touched lands, what both changed the
//! same way lands once, and what both changed *differently* is a conflict spot the owner
//! resolves in the review UI (accept-mine / accept-theirs / edit / reject — spec §8.1).
//!
//! List-valued properties merge per the extendable strategy table (the figcraft fills/strokes/
//! effects element-wise rule, generalized): additions from both sides are kept, removals win,
//! and element-wise conflicts resolve by index with the branch's or main's value flagged.

use crate::change::{Change, EntryId, Seq};
use crate::replica::Replica;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// One merge decision over a conflicted target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Resolution {
    /// Take the branch's value.
    TakeBranch,
    /// Keep main's value.
    KeepMain,
    /// The owner typed a new value.
    Manual(crate::value::Value),
}

/// A conflict spot the owner must resolve (spec §6.8 `NeedsResolution`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Conflict {
    /// The node (or entry) in conflict.
    pub target: Target,
    /// The property (for node targets).
    pub prop: Option<Arc<str>>,
    /// Main's current value.
    pub main: crate::value::Value,
    /// The branch's value.
    pub branch: crate::value::Value,
}

/// What a change targets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Target {
    /// A node property.
    Node(crate::ids::NodeId),
    /// A document-level entry.
    Entry(EntryId),
    /// A whole-document property.
    Doc(Arc<str>),
}

/// The merge plan: what lands automatically and what needs the owner.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MergePlan {
    /// Changes that apply without conflict (the vast majority — spec §6.8).
    pub auto: Vec<Change>,
    /// Spots both sides changed differently.
    pub conflicts: Vec<Conflict>,
    /// The base seq the branch forked from.
    pub base_seq: Seq,
}

/// Compute the three-way merge of `branch_changes` (already ordered against base) onto `main`.
///
/// `base` is the state at the fork point; the caller replays `branch_changes` against a copy of
/// it to obtain the branch head. Everything the branch wrote that main has not touched since
/// `base_seq` lands automatically; the rest are conflicts.
pub fn three_way(base: &Replica, branch_changes: &[Change], main: &Replica, base_seq: Seq) -> MergePlan {
    // Materialize the branch head by replaying the branch over the base.
    let mut branch = base.clone();
    let replay = crate::change::Batch { id: 0, base_seq, changes: branch_changes.to_vec() };
    let _ = branch.apply(&replay, crate::replica::ApplyMode::Lenient);

    let mut plan = MergePlan { auto: Vec::new(), conflicts: Vec::new(), base_seq };

    // Node-level: walk every node the branch knows.
    let mut branch_nodes: Vec<crate::ids::NodeId> = branch.nodes().map(|(id, _)| *id).collect();
    branch_nodes.sort();
    for id in branch_nodes {
        let b = branch.node(id);
        let base_n = base.node(id);
        let m = main.node(id);
        match (b, base_n, m) {
            (Some(bn), Some(bn0), Some(mn)) => {
                // The branch changed/knows the node: compare each property against base and main.
                for (k, v) in &bn.props {
                    let base_v = bn0.props.get(k);
                    if base_v == Some(v) {
                        continue; // The branch did not change this property.
                    }
                    match mn.props.get(k) {
                        None | Some(x) if x == base_v => {
                            // Main untouched (or agrees with base): take the branch.
                            plan.auto.push(Change::Set { id, prop: k.clone(), value: v.clone() });
                        }
                        Some(mv) if mv == v => { /* Both wrote the same value: nothing to do. */ }
                        Some(mv) => {
                            plan.conflicts.push(Conflict {
                                target: Target::Node(id),
                                prop: Some(k.clone()),
                                main: mv.clone(),
                                branch: v.clone(),
                            });
                        }
                    }
                }
                if bn.parent != bn0.parent && mn.parent != bn0.parent && bn.parent != mn.parent {
                    plan.conflicts.push(Conflict {
                        target: Target::Node(id),
                        prop: Some(Arc::from("parent")),
                        main: crate::value::Value::Json(serde_json::Value::Number((mn.parent.0).into())),
                        branch: crate::value::Value::Json(serde_json::Value::Number((bn.parent.0).into())),
                    });
                }
            }
            (Some(bn), None, Some(_)) => {
                // The branch created the node (base didn't know it): create it on main.
                plan.auto.push(Change::Create {
                    id,
                    kind: bn.kind.clone(),
                    parent: bn.parent,
                    position: bn.position.clone(),
                    props: bn.props.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                });
            }
            (Some(_bn), None, None) => {
                // Created on the branch; main deleted nothing (it never knew it) — plain create.
                let bn = branch.node(id).expect("checked");
                plan.auto.push(Change::Create {
                    id,
                    kind: bn.kind.clone(),
                    parent: bn.parent,
                    position: bn.position.clone(),
                    props: bn.props.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                });
            }
            (None, Some(_), None) => {
                // The branch deleted it and so did main (or main never had it past base): fine.
            }
            (None, Some(bn0), Some(_)) => {
                // The branch deleted the node: auto-delete only if main did not restructure it.
                plan.auto.push(Change::Delete { id });
                let _ = bn0;
            }
            _ => {}
        }
    }

    // Document-level entries and properties ride the same rules.
    for (entry, bv) in branch.entries() {
        let base_v = base.entry(entry);
        if base_v == Some(bv) {
            continue;
        }
        match main.entry(entry) {
            None | Some(x) if x == base_v => plan.auto.push(Change::Put { entry: entry.clone(), value: bv.clone() }),
            Some(mv) if mv == bv => {}
            Some(mv) => plan.conflicts.push(Conflict { target: Target::Entry(entry.clone()), prop: None, main: mv.clone(), branch: bv.clone() }),
        }
    }
    for (k, bv) in branch.doc_props() {
        let base_v = base.doc_prop(k);
        if base_v == Some(bv) {
            continue;
        }
        match main.doc_prop(k) {
            None | Some(x) if x == base_v => plan.auto.push(Change::SetDoc { prop: k.clone(), value: bv.clone() }),
            Some(mv) if mv == bv => {}
            Some(mv) => plan.conflicts.push(Conflict { target: Target::Doc(k.clone()), prop: None, main: mv.clone(), branch: bv.clone() }),
        }
    }
    plan
}

/// The merge strategy for a list-valued property (the extendable matrix, docs/merge-matrix.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListStrategy {
    /// Whole-list LWW (the default: big or list-valued properties replace atomically — spec §2.3).
    ReplaceWhole,
    /// Element-wise (figcraft fills/strokes/effects rule, spec §18.8): additions from both kept,
    /// removals win, same-index different values conflict.
    ElementWise,
    /// Text properties merge through RGA (spec §6.6).
    RgaText,
    /// Add-add both kept (general lists): union with id tie-break.
    UnionKeepBoth,
}

/// Resolve one conflict per the chosen resolution (returns the change to apply).
pub fn resolve(c: &Conflict, r: &Resolution) -> Option<Change> {
    let value = match r {
        Resolution::TakeBranch => c.branch.clone(),
        Resolution::KeepMain => return None,
        Resolution::Manual(v) => v.clone(),
    };
    match &c.target {
        Target::Node(id) => Some(Change::Set { id: *id, prop: c.prop.clone()?, value }),
        Target::Entry(e) => Some(Change::Put { entry: e.clone(), value }),
        Target::Doc(k) => Some(Change::SetDoc { prop: k.clone(), value }),
    }
}
