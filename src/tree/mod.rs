//! This file implements the tree construction.
//! This file is adopted from <https://github.com/arkworks-rs/crypto-primitives/blob/main/src/merkle_tree/mod.rs>
//! with substantial changes.
//!
//!
//!

pub mod path;
pub mod randomize_path;

use ark_std::{end_timer, start_timer};
#[cfg(feature = "parallel")]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use self::path::Path;

use super::hash::{HVCHash, LinearHash};
use crate::HEIGHT;
use core::fmt;
use std::fmt::Display;

#[derive(Clone, Debug, Default)]
pub struct Tree<H: LinearHash = HVCHash> {
    /// stores the non-leaf nodes in level order. The first element is the root node.
    /// The ith nodes (starting at 1st) children are at indices `2*i`, `2*i+1`
    non_leaf_nodes: Vec<H::Poly>,

    /// store the hash of leaf nodes from left to right
    leaf_nodes: Vec<H::Poly>,

    /// total tree height (root + interior + leaf levels). 2^(height-1) = leaf_nodes.len().
    height: usize,
}

impl<H: LinearHash> Display for Tree<H>
where
    H::Poly: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "non leaf nodes:")?;
        for (i, e) in self.non_leaf_nodes.iter().enumerate() {
            writeln!(f, "{}: {}", i, e)?;
        }
        writeln!(f, "leaf nodes:")?;
        for (i, e) in self.leaf_nodes.iter().enumerate() {
            writeln!(f, "{}: {}", i, e)?;
        }
        Ok(())
    }
}

impl<H: LinearHash> Tree<H>
where
    H: Sync,
    H::Poly: Send + Sync,
{
    /// create an empty tree of the default height (HEIGHT)
    pub fn init(hasher: &H) -> Self {
        let leaf_nodes = vec![H::Poly::default(); 1 << (HEIGHT - 1)];
        Self::new_with_leaf_nodes(&leaf_nodes, hasher)
    }

    /// create a new tree with leaf nodes; height is derived from leaf count
    /// (which must be a power of two ≥ 2).
    pub fn new_with_leaf_nodes(leaf_nodes: &[H::Poly], hasher: &H) -> Self {
        let timer = start_timer!(|| format!("generate new tree with {} leaves", leaf_nodes.len()));

        let len = leaf_nodes.len();
        assert!(len >= 2 && len.is_power_of_two(), "leaf count must be a power of two ≥ 2");
        let height = len.trailing_zeros() as usize + 1;

        let mut non_leaf_nodes = vec![H::Poly::default(); len - 1];

        // Compute the starting indices for each non-leaf level of the tree
        let mut index = 0;
        let mut level_indices = Vec::with_capacity(height - 1);
        for _ in 0..(height - 1) {
            level_indices.push(index);
            index = left_child_index(index);
        }

        // compute the hash values for the non-leaf bottom layer
        {
            let start_index = level_indices.pop().unwrap();
            let upper_bound = left_child_index(start_index);

            #[cfg(feature = "parallel")]
            let bottom_iter = non_leaf_nodes.par_iter_mut().enumerate();
            #[cfg(not(feature = "parallel"))]
            let bottom_iter = non_leaf_nodes.iter_mut().enumerate();
            bottom_iter
                .take(upper_bound)
                .skip(start_index)
                .for_each(|(current_index, e)| {
                    // `left_child_index(current_index)` and `right_child_index(current_index) returns the position of
                    // leaf in the whole tree (represented as a list in level order). We need to shift it
                    // by `-upper_bound` to get the index in `leaf_nodes` list.
                    let left_leaf_index = left_child_index(current_index) - upper_bound;
                    let right_leaf_index = right_child_index(current_index) - upper_bound;
                    // compute hash
                    *e = hasher.decom_then_hash(
                        &leaf_nodes[left_leaf_index],
                        &leaf_nodes[right_leaf_index],
                    );
                });
        }

        // compute the hash values for nodes in every other layer in the tree
        level_indices.reverse();

        for &start_index in &level_indices {
            // The layer beginning `start_index` ends at `upper_bound` (exclusive).
            let upper_bound = left_child_index(start_index);
            let mut buf = non_leaf_nodes[start_index..upper_bound].to_vec();
            #[cfg(feature = "parallel")]
            let upper_iter = buf.par_iter_mut().enumerate();
            #[cfg(not(feature = "parallel"))]
            let upper_iter = buf.iter_mut().enumerate();
            upper_iter.for_each(|(index, node)| {
                *node = hasher.decom_then_hash(
                    &non_leaf_nodes[left_child_index(index + start_index)],
                    &non_leaf_nodes[right_child_index(index + start_index)],
                )
            });
            non_leaf_nodes[start_index..upper_bound].clone_from_slice(buf.as_ref());
        }
        end_timer!(timer);
        Self {
            non_leaf_nodes,
            leaf_nodes: leaf_nodes.to_vec(),
            height,
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn root(&self) -> H::Poly {
        self.non_leaf_nodes[0]
    }

    // generate a membership proof for the given index
    pub fn gen_proof(&self, index: usize) -> Path<H> {
        let timer = start_timer!(|| "generate membership proof");
        // Get Leaf hash, and leaf sibling hash,
        let leaf_index_in_tree = convert_index_to_last_level(index, self.height);

        // path.len() = `tree height - 1`, the missing elements being the root
        let mut nodes = Vec::with_capacity(self.height - 1);
        if index.is_multiple_of(2) {
            nodes.push((self.leaf_nodes[index], self.leaf_nodes[index + 1]))
        } else {
            nodes.push((self.leaf_nodes[index - 1], self.leaf_nodes[index]))
        }

        // Iterate from the bottom layer after the leaves, to the top, storing all nodes and their siblings.
        let mut current_node = parent_index(leaf_index_in_tree).unwrap();
        while current_node != 0 {
            let sibling_node = sibling_index(current_node).unwrap();
            if is_left_child(current_node) {
                nodes.push((
                    self.non_leaf_nodes[current_node],
                    self.non_leaf_nodes[sibling_node],
                ));
            } else {
                nodes.push((
                    self.non_leaf_nodes[sibling_node],
                    self.non_leaf_nodes[current_node],
                ));
            }
            current_node = parent_index(current_node).unwrap();
        }

        // we want to make path from root to bottom
        nodes.reverse();
        end_timer!(timer);
        Path { index, nodes }
    }
}

/// Returns the index of the sibling, given an index.
#[inline]
fn sibling_index(index: usize) -> Option<usize> {
    if index == 0 {
        None
    } else if is_left_child(index) {
        Some(index + 1)
    } else {
        Some(index - 1)
    }
}

/// Returns the index of the parent, given an index.
#[inline]
fn parent_index(index: usize) -> Option<usize> {
    if index > 0 {
        Some((index - 1) >> 1)
    } else {
        None
    }
}

/// Returns the index of the left child, given an index.
#[inline]
fn left_child_index(index: usize) -> usize {
    2 * index + 1
}

/// Returns the index of the right child, given an index.
#[inline]
fn right_child_index(index: usize) -> usize {
    2 * index + 2
}

#[inline]
fn convert_index_to_last_level(index: usize, tree_height: usize) -> usize {
    index + (1 << (tree_height - 1)) - 1
}

/// Returns true iff the given index represents a left child.
#[inline]
fn is_left_child(index: usize) -> bool {
    index % 2 == 1
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::poly::Polynomial;
    use crate::HVCPoly;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    #[test]
    fn test_tree() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let hasher = HVCHash::init(&mut rng);

        let leafs: Vec<HVCPoly> = (0..(1 << (HEIGHT - 1)))
            .map(|_| HVCPoly::rand_poly(&mut rng))
            .collect();
        let tree = Tree::new_with_leaf_nodes(&leafs, &hasher);

        for _ in 0..100 {
            let index = rng.next_u32() % (1 << (HEIGHT - 1));
            let proof = tree.gen_proof(index as usize);
            assert!(proof.verify(&tree.non_leaf_nodes[0], &hasher));
        }
    }

    #[test]
    fn test_tree_dynamic_height() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let hasher = HVCHash::init(&mut rng);

        for log_leaves in 1..=6 {
            let n_leaves = 1usize << log_leaves;
            let leafs: Vec<HVCPoly> = (0..n_leaves).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
            let tree = Tree::new_with_leaf_nodes(&leafs, &hasher);
            assert_eq!(tree.height(), log_leaves + 1);
            for index in 0..n_leaves {
                let proof = tree.gen_proof(index);
                assert!(proof.verify(&tree.root(), &hasher), "verify failed at log_leaves={log_leaves}, index={index}");
            }
        }
    }
}
