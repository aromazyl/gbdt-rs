//
// tree.rs
// Copyright (C) 2019 zhangyule <zyl2336709@gmail.com>
// Distributed under terms of the MIT license.
//

use data::*;

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
                self.nodes[i].stats[j].s = 0;
                self.nodes[i].stats[j].min_delta_loss = 99999999.0;
                self.nodes[i].stats[j].fea_value = 0;
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
        }
    }

}
