#pragma once

#include "duckdb/optimizer/predicate_transfer/table_operator_namager.hpp"
#include "duckdb/optimizer/predicate_transfer/dag.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

class EdgeInfo {
public:
	EdgeInfo(const LogicalType &type, LogicalOperator &left, const ColumnBinding &left_binding, LogicalOperator &right,
	         const ColumnBinding &right_binding)
	    : return_type(type), left_table(left), left_binding(left_binding), right_table(right),
	      right_binding(right_binding) {
	}

	LogicalType return_type;

	LogicalOperator &left_table;
	ColumnBinding left_binding;

	LogicalOperator &right_table;
	ColumnBinding right_binding;

	bool protect_left = false;
	bool protect_right = false;
};

class TransferGraphManager {
public:
	explicit TransferGraphManager(ClientContext &context) : context(context), table_operator_manager(context) {
	}

	ClientContext &context;
	TableOperatorManager table_operator_manager;
	TransferGraph transfer_graph;
	vector<LogicalOperator *> transfer_order;

public:
	bool Build(LogicalOperator &op);
	void AddFilterPlan(idx_t create_table, const shared_ptr<FilterPlan> &filter_plan, bool reverse);

private:
	void ExtractEdgesInfo(const vector<reference<LogicalOperator>> &join_operators);
	// void CreatePineTree();
	void CreatePredicateTransferGraph();
	void LargestRoot(vector<LogicalOperator *> &sorted_nodes);

	pair<idx_t, idx_t> FindEdge(const unordered_set<idx_t> &constructed_set,
	                            const unordered_set<idx_t> &unconstructed_set);

private:
	void IgnoreUnfilteredTable();

private:
	unordered_map<idx_t, unordered_map<idx_t, shared_ptr<EdgeInfo>>> neighbor_matrix;
	vector<shared_ptr<EdgeInfo>> selected_edges;

	// Join Key Table Groups
	// Todo: replace set with unordered set
	// struct HashFunc {
	// 	size_t operator()(const ColumnBinding &key) const {
	// 		return std::hash<uint64_t> {}(key.table_index) ^ (std::hash<uint64_t> {}(key.column_index) << 1);
	// 	}
	// };
	// unordered_map<ColumnBinding, shared_ptr<set<idx_t>>, HashFunc> join_keys_table_groups;
};
} // namespace duckdb
