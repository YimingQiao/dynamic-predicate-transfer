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
	void CreateTransferPlan();
	void LargestRoot(vector<LogicalOperator *> &sorted_nodes);

	pair<idx_t, idx_t> FindEdge(const unordered_set<idx_t> &constructed_set,
	                            const unordered_set<idx_t> &unconstructed_set);

private:
	//! Classify all tables into three categories: intermediate table, unfiltered table, and filtered table.
	void ClassifyTables();
	void SkipUnfilteredTable();

	// Join Key Table Groups
	struct JoinKeyTableGroup {
		LogicalType return_type;
		unordered_set<idx_t> table_ids;

		JoinKeyTableGroup(const LogicalType &return_type, const idx_t table_id) : return_type(return_type) {
			table_ids.insert(table_id);
		}

		void Union(const JoinKeyTableGroup &other) {
			D_ASSERT(return_type == other.return_type);
			table_ids.insert(other.table_ids.begin(), other.table_ids.end());
		}
	};

	struct HashFunc {
		size_t operator()(const ColumnBinding &key) const {
			return std::hash<uint64_t> {}(key.table_index) ^ (std::hash<uint64_t> {}(key.column_index) << 1);
		}
	};

	//! From table id to its join keys
	unordered_map<idx_t, vector<ColumnBinding>> table_join_keys;
	//! From join keys to its table groups
	unordered_map<ColumnBinding, shared_ptr<JoinKeyTableGroup>, HashFunc> table_groups;
    //! Table categories
	unordered_set<idx_t> unfiltered_table;
	unordered_set<idx_t> filtered_table;
	unordered_set<idx_t> intermediate_table;

private:
	unordered_map<idx_t, unordered_map<idx_t, shared_ptr<EdgeInfo>>> neighbor_matrix;
	vector<shared_ptr<EdgeInfo>> selected_edges;
};
} // namespace duckdb
