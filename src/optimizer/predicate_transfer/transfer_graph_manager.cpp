#include "duckdb/optimizer/predicate_transfer/transfer_graph_manager.hpp"

#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/optimizer/predicate_transfer/predicate_transfer_optimizer.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include <queue>

namespace duckdb {

bool TransferGraphManager::Build(LogicalOperator &plan) {
	// 1. Extract all operators, including table operators and join operators
	const vector<reference<LogicalOperator>> joins = table_operator_manager.ExtractOperators(plan);
	if (table_operator_manager.table_operators.size() < 2) {
		return false;
	}

	// 2. Getting graph edges information from join operators
	ExtractEdgesInfo(joins);
	if (neighbor_matrix.empty()) {
		return false;
	}

	// 2.5 Remove Bloom filter creation for unfiltered tables
	IgnoreUnfilteredTable();

	// 3. Create the transfer graph
	CreatePredicateTransferGraph();
	return true;
}

void TransferGraphManager::AddFilterPlan(idx_t create_table, const shared_ptr<FilterPlan> &filter_plan, bool reverse) {
	bool is_forward = !reverse;

	D_ASSERT(!filter_plan->apply.empty());
	auto &expr = filter_plan->apply[0];
	auto node_idx = expr.table_index;
	transfer_graph[node_idx]->Add(create_table, filter_plan, is_forward, true);
}

void TransferGraphManager::ExtractEdgesInfo(const vector<reference<LogicalOperator>> &join_operators) {
	unordered_set<hash_t> existed_set;
	auto ComputeConditionHash = [](const JoinCondition &cond) {
		return cond.left->Hash() + cond.right->Hash();
	};

	for (size_t i = 0; i < join_operators.size(); i++) {
		auto &join = join_operators[i].get();
		if (join.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
		    join.type != LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			continue;
		}

		auto &comp_join = join.Cast<LogicalComparisonJoin>();
		D_ASSERT(comp_join.expressions.empty());

		for (size_t j = 0; j < comp_join.conditions.size(); j++) {
			auto &cond = comp_join.conditions[j];
			if (cond.comparison != ExpressionType::COMPARE_EQUAL ||
			    cond.left->type != ExpressionType::BOUND_COLUMN_REF ||
			    cond.right->type != ExpressionType::BOUND_COLUMN_REF) {
				continue;
			}

			hash_t hash = ComputeConditionHash(cond);
			if (!existed_set.insert(hash).second) {
				continue;
			}

			auto &left_col_expression = cond.left->Cast<BoundColumnRefExpression>();
			ColumnBinding left_binding = table_operator_manager.GetRenaming(left_col_expression.binding);
			auto left_node = table_operator_manager.GetTableOperator(left_binding.table_index);

			auto &right_col_expression = cond.right->Cast<BoundColumnRefExpression>();
			ColumnBinding right_binding = table_operator_manager.GetRenaming(right_col_expression.binding);
			auto right_node = table_operator_manager.GetTableOperator(right_binding.table_index);

			if (!left_node || !right_node) {
				continue;
			}

			// Create edge
			auto edge =
			    make_shared_ptr<EdgeInfo>(cond.left->return_type, *left_node, left_binding, *right_node, right_binding);

			// Set protection flags
			switch (comp_join.type) {
			case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
				if (comp_join.join_type == JoinType::LEFT) {
					edge->protect_left = true;
				} else if (comp_join.join_type == JoinType::MARK) {
					edge->protect_right = true;
				} else if (comp_join.join_type == JoinType::RIGHT) {
					edge->protect_right = true;
				} else if (comp_join.join_type != JoinType::INNER && comp_join.join_type != JoinType::SEMI &&
				           comp_join.join_type != JoinType::RIGHT_SEMI) {
					continue; // Unsupported join type
				}
				break;
			}
			case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
				if (comp_join.delim_flipped == 0) {
					edge->protect_left = true;
				} else {
					edge->protect_right = true;
				}
				break;
			}
			default:
				continue;
			}

			// Store edge info
			neighbor_matrix[left_binding.table_index][right_binding.table_index] = edge;
			neighbor_matrix[right_binding.table_index][left_binding.table_index] = edge;

			// The left table and right table belong to the same table group
			if (!edge->protect_left && !edge->protect_right) {
				auto &group_left = table_groups[left_binding];
				if (group_left == nullptr) {
					group_left =
					    make_shared_ptr<JoinKeyTableGroup>(left_col_expression.return_type, left_binding.table_index);
				}

				auto &group_right = table_groups[right_binding];
				if (group_right == nullptr) {
					group_right =
					    make_shared_ptr<JoinKeyTableGroup>(right_col_expression.return_type, right_binding.table_index);
				}

				// union groups
				group_left->Union(*group_right);
				table_groups[right_binding] = group_left;
			}
		}
	}
}

void TransferGraphManager::ClassifyTables() {
	for (auto &pair : table_operator_manager.table_operators) {
		auto id = pair.first;
		auto &table = pair.second;
		auto &edges = neighbor_matrix[id];

		// Check intermediate table, which belongs to more than 2 groups
		bool is_intermediate = false;
		unordered_set<JoinKeyTableGroup *> belong_groups;
		for (auto &sub_pair : edges) {
			auto &edge = sub_pair.second;
			auto &join_key_group = table_groups[edge->left_binding];
			belong_groups.insert(join_key_group.get());

			if (belong_groups.size() > 1) {
				is_intermediate = true;
				intermediate_table.insert(id);
				break;
			}
		}
		if (is_intermediate) {
			continue;
		}

		// Check unfiltered table
		if (table->type == LogicalOperatorType::LOGICAL_GET) {
			auto &get = table->Cast<LogicalGet>();
			if (get.table_filters.filters.empty()) {
				unfiltered_table.insert(id);
				continue;
			}
		}

		// Last, it is a filtered table
		filtered_table.insert(id);
	}
}

void TransferGraphManager::SkipUnfilteredTable() {
	// 1. Classify Tables
	ClassifyTables();

	bool changed = false;
	do {
		changed = false;
		for (auto &table_idx : unfiltered_table) {
			// 2.1 collect received bfs
			unordered_map<idx_t, vector<shared_ptr<EdgeInfo>>> received_bfs;
			auto &edges = neighbor_matrix[table_idx];

			for (auto &pair : edges) {
				auto &e = pair.second;

				if (e->left_binding.table_index == table_idx && !e->protect_left) {
					auto &bfs = received_bfs[e->left_binding.column_index];
					bfs.push_back(e);
				} else if (e->right_binding.table_index == table_idx && !e->protect_right) {
					auto &bfs = received_bfs[e->right_binding.column_index];
					bfs.push_back(e);
				}
			}

			// 2.2 remove BFs creation for this table
			for (auto &pair : edges) {
				auto &edge = pair.second;

				bool is_left = (edge->left_binding.table_index == table_idx && !edge->protect_right);
				bool is_right = (edge->right_binding.table_index == table_idx && !edge->protect_left);
				if (!is_left && !is_right) {
					continue;
				}

				idx_t col_idx = is_left ? edge->left_binding.column_index : edge->right_binding.column_index;
				auto &bfs = received_bfs[col_idx];

				// 2.2.1 add new links
				for (auto &bf_link : bfs) {
					// the same edge
					if (bf_link->left_binding == edge->left_binding && bf_link->right_binding == edge->right_binding) {
						continue;
					}

					bool bf_left = (bf_link->left_binding.table_index == table_idx && !bf_link->protect_left);
					bool bf_right = (bf_link->right_binding.table_index == table_idx && !bf_link->protect_right);
					if (!bf_left && !bf_right) {
						continue;
					}

					shared_ptr<EdgeInfo> concat_edge = nullptr;
					if (is_left && bf_left) {
						concat_edge =
						    make_shared_ptr<EdgeInfo>(edge->return_type, bf_link->right_table, bf_link->right_binding,
						                              edge->right_table, edge->right_binding);
						concat_edge->protect_left = true;
					} else if (is_left && bf_right) {
						concat_edge =
						    make_shared_ptr<EdgeInfo>(edge->return_type, bf_link->left_table, bf_link->left_binding,
						                              edge->right_table, edge->right_binding);
						concat_edge->protect_left = true;
					} else if (is_right && bf_left) {
						concat_edge = make_shared_ptr<EdgeInfo>(edge->return_type, edge->left_table, edge->left_binding,
						                                        bf_link->right_table, bf_link->right_binding);
						concat_edge->protect_right = true;
					} else if (is_right && bf_right) {
						concat_edge = make_shared_ptr<EdgeInfo>(edge->return_type, edge->left_table, edge->left_binding,
						                                        bf_link->left_table, bf_link->left_binding);
						concat_edge->protect_right = true;
					}

					if (concat_edge) {
						idx_t i = concat_edge->left_binding.table_index;
						idx_t j = concat_edge->right_binding.table_index;
						auto &edge_ij = neighbor_matrix[i][j];
						auto &edge_ji = neighbor_matrix[j][i];

						bool exists = false;
						if (edge_ij != nullptr) {
							bool same_direction = edge_ij->left_binding == concat_edge->left_binding &&
							                      edge_ij->right_binding == concat_edge->right_binding;
							bool reverse_direction = edge_ij->left_binding == concat_edge->right_binding &&
							                         edge_ij->right_binding == concat_edge->left_binding;

							if (same_direction || reverse_direction) {
								if (same_direction) {
									edge_ij->protect_left &= concat_edge->protect_left;
									edge_ij->protect_right &= concat_edge->protect_right;
								} else { // reverse_direction
									edge_ij->protect_left &= concat_edge->protect_right;
									edge_ij->protect_right &= concat_edge->protect_left;
								}
								exists = true;
							}
						}

						if (!exists) {
							edge_ij = concat_edge;
							edge_ji = concat_edge;
						}
					}
				}

				// 2.2.2 disable current link
				if (is_left) {
					edge->protect_right = true;
				} else {
					edge->protect_left = true;
				}

				changed = true;
			}

			// 2.3. Remove invalid links
			for (auto it = edges.begin(); it != edges.end();) {
				auto &edge = it->second;

				// If the condition is met, erase the item from the unordered_map
				if (edge->protect_left && edge->protect_right) {
					it = edges.erase(it);
				} else {
					++it;
				}
			}
		}
	} while (changed);
}

void TransferGraphManager::LargestRoot(vector<LogicalOperator *> &sorted_nodes) {
	unordered_set<idx_t> constructed_set, unconstructed_set;
	int prior_flag = static_cast<int>(table_operator_manager.table_operators.size()) - 1;
	idx_t root = std::numeric_limits<idx_t>::max();

	// Initialize nodes
	for (auto &entry : table_operator_manager.table_operators) {
		idx_t id = entry.first;
		auto node = make_uniq<GraphNode>(id, prior_flag--);

		if (entry.second == sorted_nodes.back()) {
			root = id;
			constructed_set.insert(id);
		} else {
			unconstructed_set.insert(id);
		}

		transfer_graph[id] = std::move(node);
	}

	// Add root
	transfer_order.push_back(table_operator_manager.GetTableOperator(root));
	table_operator_manager.table_operators.erase(root);

	// Build graph
	while (!unconstructed_set.empty()) {
		auto selected_edge = FindEdge(constructed_set, unconstructed_set);
		if (selected_edge.first == std::numeric_limits<idx_t>::max()) {
			break;
		}

		auto &edge = neighbor_matrix[selected_edge.first][selected_edge.second];
		selected_edges.emplace_back(std::move(edge));

		auto node = transfer_graph[selected_edge.second].get();
		node->cardinality_order = prior_flag--;

		transfer_order.push_back(table_operator_manager.GetTableOperator(node->id));
		table_operator_manager.table_operators.erase(node->id);
		unconstructed_set.erase(selected_edge.second);
		constructed_set.insert(selected_edge.second);
	}
}

void TransferGraphManager::CreateTransferPlan() {
	auto saved_nodes = table_operator_manager.table_operators;
	while (!table_operator_manager.table_operators.empty()) {
		LargestRoot(table_operator_manager.sorted_table_operators);
		table_operator_manager.SortTableOperators();
	}
	table_operator_manager.table_operators = saved_nodes;

	for (auto &edge : selected_edges) {
		if (!edge) {
			continue;
		}

		idx_t left_idx = TableOperatorManager::GetScalarTableIndex(&edge->left_table);
		idx_t right_idx = TableOperatorManager::GetScalarTableIndex(&edge->right_table);

		D_ASSERT(left_idx != std::numeric_limits<idx_t>::max() && right_idx != std::numeric_limits<idx_t>::max());

		auto &type = edge->return_type;
		auto left_node = transfer_graph[right_idx].get();
		auto right_node = transfer_graph[left_idx].get();

		auto &left_cols = edge->left_binding;
		auto &right_cols = edge->right_binding;

		auto protect_left = edge->protect_left;
		auto protect_right = edge->protect_right;

		// smaller table is in the left
		if (left_node->cardinality_order > right_node->cardinality_order) {
			std::swap(left_node, right_node);
			std::swap(left_cols, right_cols);
			std::swap(protect_left, protect_right);
		}

		// forward: from the smaller to the larger
		if (!protect_left) {
			left_node->Add(right_node->id, {left_cols}, {right_cols}, {type}, true, false);
			right_node->Add(left_node->id, {left_cols}, {right_cols}, {type}, true, true);
		}

		// backward: from the larger to the smaller
		if (!protect_right) {
			left_node->Add(right_node->id, {left_cols}, {right_cols}, {type}, false, true);
			right_node->Add(left_node->id, {left_cols}, {right_cols}, {type}, false, false);
		}
	}
}

pair<idx_t, idx_t> TransferGraphManager::FindEdge(const unordered_set<idx_t> &constructed_set,
                                                  const unordered_set<idx_t> &unconstructed_set) {
	pair<idx_t, idx_t> result {std::numeric_limits<idx_t>::max(), std::numeric_limits<idx_t>::max()};
	idx_t max_cardinality = 0;

	for (auto i : unconstructed_set) {
		for (auto j : constructed_set) {
			auto &edges = neighbor_matrix[j][i];
			if (edges == nullptr) {
				continue;
			}

			idx_t cardinality = table_operator_manager.GetTableOperator(i)->estimated_cardinality;
			if (cardinality > max_cardinality) {
				max_cardinality = cardinality;
				result = {j, i};
			}
		}
	}
	return result;
}

} // namespace duckdb
