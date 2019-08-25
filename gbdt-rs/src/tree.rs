//
// tree.rs
// Copyright (C) 2019 zhangyule <zyl2336709@gmail.com>
// Distributed under terms of the MIT license.
//

use data::*;
use num::pow;

lazy_static! {
    static ref TRIVALSPLITVALUE: f64 = -999999;
}

#[derive(Copy, Clone, Debug)]
struct TreeStatsNode {
    s: f64,
    min_delta_loss: f64,
    fea_value: f64,
    m_s: f64,
    l_s: f64,
}

struct TreeNode {
    left_child_index: i32,
    right_child_index: i32,
    feature_id: i32,
    split_value: f64,
    label: f32,
    m_inf: i32,
    l_inf: f64,
    stats: Vec<TreeStatsNode>,
    gain: f64,
}

#[derive(Debug)]
struct Tree {
    depth: i32,
    node_num: u32,
    doc_num: u32,
    fea_num: u32,
    nodes: Vec<TreeNode>,
    node_idx: Vec<u32>,
    fx: Vec<f64>,
    g: Vec<f64>,
    fea_significance: Vec<f64>,
}

trait TreeMethod {
    fn new(depth: u32, doc_num: u32, num_fea: u32) -> Self
    fn reset(&mut self);
    fn initFx(&mut self, data: &Data);
    fn updateFx(&mut self, data: &Data);
    fn calculateGradient(&mut self, data: &Data);
    fn calculateCoefficient(&mut self, data: &Data);
    fn generateTreeStructure(&mut self, data: &Data);
    fn findBestSplitValueOneFeature(&mut self, data: &Data, currentDepth: &i32, feaId: &i32);
    fn split(&mut self, data: &Data, currentDepth: &i32);
}

impl TreeMethod for Tree {
    fn new(depth: u32, doc_num: u32, num_fea: u32) -> Self {
        let mut fx = Vec::with_capacity(doc_num);
        fx.resize(doc_num, 0);
        let mut g = Vec::with_capacity(doc_num);
        g.resize(doc_num, 0);
        let node_num = ((1 << depth) - 1) as i32;
        let mut nodes: Vec<TreeNode> = Vec::with_capacity((1 << depth) - 1);
        for i in 0..node_nums {
            let left_child: i32;
            let right_child: i32;
            if i < node_nums / 2 {
                left_child = i * 2 + 1;
                right_child = i * 2 - 1;
            } else {
                left_child = -1;
                right_child = -1;
            }
            let mut stats = Vec<TreeStatsNode>::new();
            for j in 0..fea_num {
                let stat = TreeStatsNode {
                    s: 0.0,
                    min_delta_loss: 99999.0,
                    fea_value: 0,
                    m_s = 0,
                    l_s = 0,
                };
                stats.push(stat);
            }
            nodes.push(TreeNode {
                left_child_index: left_child,
                right_child_index: right_child,
                feature_id: 0,
                split_value: 0.0f64,
                label: 0,
                m_inf: 0,
                l_inf: 0,
                stats,
                gain: 0,
            });
        }
        let fx = Vec::with_capacity(doc_num);
        fx.resize(doc_num, 0);
        let gx = Vec::with_capacity(doc_num);
        gx.resize(doc_num, 0);
        let fea_significance = Vec::with_capacity(fea_num);
        let mut node_idx = Vec::with_capacity(doc_num);
        node_idx.resize(doc_num, 0);
        Tree {
            depth,
            node_num,
            doc_num,
            fea_num,
            nodes,
            node_idx,
            fx,
            gx,
            fea_significance,
        }
    }

    fn reset(&mut self) {
        self.node_idx.resize(self.doc_num, 0);
        for i in 0..self.node_num {
            self.nodes[i].feature_id = -1;
            self.nodes[i].split_value = 0;
            self.nodes[i].label = 0;
            self.nodes[i].m_inf = 0;
            self.nodes[i].l_inf = 0;
            self.nodes[i].gain = 0;
            for j in 0..self.fea_num {
                self.nodes[i].stats[j].s = TRIVALSPLITVALUE;
                self.nodes[i].stats[j].min_delta_loss = 99999999.0;
                self.nodes[i].stats[j].fea_value = TRIVALSPLITVALUE;
                self.nodes[i].stats[j].m_s = 0;
                self.nodes[i].stats[j].l_s = 0;
            }
        }
    }
    fn initFx(&mut self, _data: &Data) {}
    fn updateFx(&mut self, data: &Data) {
        for i in 0..self.doc_num {
            let idx = self.node_idx[i];
            if (idx > 0) {
                self.fx[i] += self.nodes[idx].label;
            }
        }
    }

    fn generateTreeStructure(&mut self, data: &Data) {
        self.reset();
        for depth in 1..self.depth {
            self.split(data, depth);
        }
    }

    fn split(&mut self, data: &Data, currentDepth: &i32) {
        if currentDepth >= self.depth || currentDepth < 0 {
            return;
        }
        let idx_start = (1 << (currentDepth - 1)) - 1;
        let idx_end = (1 << (currentDepth)) - 2;

        if idx_start == idx_end {
            let ref mut tnode = self.nodes[0];
            for i in i..self.doc_num {
                if self.node_idx[i] >= 0 {
                    tnode.m_inf += 1;
                    tnode.l_inf += self.g[i];
                }
            }
            tnode.label = tnode.l_inf / tnode.m_inf;
        }

        for fea_id in data.min_feature_idx .. data.max_feature_idx {
            self.FindBestSplitValueOneFeature(&data, &currentDepth, &fea_id);
        }

        for fea_id in idx_start..idx_end {
            let ref mut tnode = self.nodes[i];
            if tnode.m_inf <= 10 {
                tnode.feature_id = -1;
                tnode.split_value = 0;
                continue;
            }
            let mut min_obj: f64 = 99999999.0;
            if tnode.stats[fea_id].min_delta_loss < min_obj {
                min_obj = tnode.stats[fea_id].min_delta_loss;
                tnode.split_value = tnode.stats[fea_id].s;
                tnode.feature_id = fea_id;
            }
        }

        let mut tree_node_idx = -1;

        for i in 0..self.doc_num {
            tree_node_idx = self.node_idx[i];
            if tree_node_idx < idx_start || tree_node_idx > idx_end {
                continue;
            }
            let ref tnode = self.nodes[tree_node_idx];
            if tnode.fea_id < 0 {
                continue;
            }
            let next_idx: i32;
            if data.raw_data[i].values[tnode.fea_id] < tnode.split_value {
                next_idx = tnode.left_child_idx;
            } else {
                next_idx = tnode.right_child_index;
            }
            self.nodes[next_idx as usize].m_inf += 1;
            self.nodes[next_idx as usize].l_inf += self.g[i];
            self.node_idx[i] = next_idx;
        }
        for i in idx_start..idx_end {
            let ref tnode = self.nodes[i as usize];
            if tnode.fea_id < 0 {
                continue
            }
            let mut ref left_child_node = self.nodes[tnode.left_child_index];
            let mut ref right_child_node = self.nodes[tnode.right_child_index];
            left_child_node.label = 0;
            if left_child_node.m_inf > 0 {
                left_child_node.label = left_child_node.l_inf / left_child_node.m_inf;
            }
            right_child_node.label = 0;
            if right_child_node.m_inf > 0 {
                right_child_node.label = right_child_node.l_inf / right_child_node.m_inf;
            }
            tnode.gain = left_child_node.m_inf * right_child_node.m_inf / tnode.m_inf;
            tnode.gain *= pow(left_child_node.label - right_child_node.label);
            self.fea_significance[tnode.feature_id] += tnode.gain;
        }
        return true
    }

    fn FindBestSplitValueOneFeature(&mut self, data: &Data, currentDepth: &i32, feaId: &i32) {
        let idx_start: usize = ((currentDepth - 1) << 1) - 1;
        let idx_end: usize = ((currentDepth) << 1) - 2;
        let ref sorted_fea_vec = data.sorted_data[feaId];

        let mut tree_node_idx = -1;
        for i in 0..self.doc_num {
            let ref snode = sorted_fea_vec[i];
            tree_node_idx = self.node_idx[snode.doc_id];
            if tree_node_idx < idx_start || tree_node_idx > idx_end {
                continue;
            }
            let mut ref tnode = self.nodes[tree_node_idx];
            let mut ref stats_node = tnode.stats[feaId];
            if stats_node.fea_value != snode.value {
                if TRIVALSPLITVALUE == stats_node.fea_value {
                    stats_node.s = snode.value;
                    stats_node.min_delta_loss = - pow(tnodel_inf, 2) / tnode.m_inf;
                } else {
                    let v = -pow(stats_node.l_s, 2) / stats_node.m_s - pow(tnode.l_inf - stats_node.l_s, 2) / (t_node.m_inf - stats_node.m_s);
                    if v < stats_node.min_delta_loss {
                        stats_node.s = (stats_node.fea_value ++ snode.value) / 2.0;
                        stats_node.min_delta_loss = v;
                    }
                }
                stats_node.fea_value = snode.value;
            }
            stats_node.m_s += 1;
            stats_node.l_s += g_[snode.doc_id];
        }
    }
}

