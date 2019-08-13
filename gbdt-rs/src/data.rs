//
// data.rs
// Copyright (C) 2019 zhangyule <zyl2336709@gmail.com>
// Distributed under terms of the MIT license.
//

// #[macro_use] extern crate scan_fmt;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::str::FromStr;


struct DataNode {
    id: i32,
    ground_truth: f32,
    values: Vec<f32>,
}

#[derive(Copy, Clone)]
struct SortedDataNode {
    doc_id: i32,
    value: f32,
}

struct Data {
    raw_data: Vec<DataNode>,
    sorted_data: Vec<Vec<SortedDataNode>>,
    min_fea_idx: u32,
    max_fea_idx: u32,
    fea_num: u32,
    doc_num: u32,
}

impl Data {
    fn new(filename: String, doc_num: u32, fea_num: u32) -> Self {
        let mut min_fea_idx = 0u32;
        let mut max_fea_idx = 0u32;
        let f = File::open(filename).unwrap();
        let f = BufReader::new(f);
        let raw_data: Vec<DataNode>
            = Vec::with_capacity(doc_num as usize);
        let mut sorted_data: Vec<Vec<SortedDataNode>>
            = Vec::with_capacity(fea_num as usize);

        let mut doc_idx = -1;
        for _ in 0..fea_num {
            sorted_data.push(Vec::with_capacity(doc_num as usize));
        }
        for line in f.lines() {
            doc_idx += 1;
            let line = line.unwrap();
            let terms = line.split_whitespace().collect::<Vec<&str>>();
            let label = f32::from_str(terms[1]).unwrap();
            let mut data_node = DataNode {
                id: doc_idx,
                ground_truth: (2.0 * label - 1.0),
                values: Vec::with_capacity(fea_num as usize),
            };
            for fidx in 0..fea_num {
                data_node.values.push(0.0);
                sorted_data[fidx as usize].push(SortedDataNode {
                    doc_id: doc_idx,
                    value: -9999.0,
                });
            }

            let mut iter = terms.iter();
            iter.next();
            iter.next();

            for term in iter {
                let (feature_index, feature_value) = scan_fmt!(term, "{d}:{f}", u32, f32).unwrap();
                data_node.values[feature_index as usize] = feature_value;
                sorted_data[feature_index as usize][doc_idx as usize].value = feature_value;
                if feature_index > max_fea_idx {
                    max_fea_idx = feature_index;
                }
                if feature_index < min_fea_idx {
                    min_fea_idx = feature_index;
                }
            }
            assert_eq!(doc_idx, doc_num as i32, "the input doc_num {} must be equal to real doc_num {}",
                       doc_num, doc_idx);
            for i in min_fea_idx..max_fea_idx {
                if i % 10  == 0 {
                    println!("Sorting feature: {}", i);
                }
                sorted_data[i as usize].sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
            }
        }
        Data {
            raw_data,
            sorted_data,
            min_fea_idx,
            max_fea_idx,
            fea_num,
            doc_num,
        }
    }
}
