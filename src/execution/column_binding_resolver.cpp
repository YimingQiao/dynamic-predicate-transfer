#include "duckdb/execution/column_binding_resolver.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_recursive_cte.hpp"
#include "duckdb/planner/operator/logical_create_bf.hpp"
#include "duckdb/planner/operator/logical_use_bf.hpp"

namespace duckdb {

ColumnBindingResolver::ColumnBindingResolver(bool verify_only) : verify_only(verify_only) {
}

void ColumnBindingResolver::VisitOperator(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_CREATE_BF: {
		VisitOperatorChildren(op);
		auto &create_bf = op.Cast<LogicalCreateBF>();
		for (auto &bf_plan : create_bf.filter_plans) {
			bf_plan->bound_cols_build.clear();
			for (auto &col : bf_plan->build) {
				auto &col_bind = col;
				for (idx_t i = 0; i < bindings.size(); i++) {
					if (col_bind == bindings[i]) {
						bf_plan->bound_cols_build.push_back(i);
						break;
					}
				}
			}
			if (bf_plan->bound_cols_build.size() != bf_plan->build.size()) {
				throw InternalException("Predicate Transfer: Failed to bind column reference.\n");
			}
		}
		bindings = op.GetColumnBindings();
		VisitOperatorExpressions(op);
		return;
	}
	case LogicalOperatorType::LOGICAL_USE_BF: {
		VisitOperatorChildren(op);
		auto &use_bf = op.Cast<LogicalUseBF>();
		auto &bf_plan = use_bf.filter_plan;
		bf_plan->bound_cols_apply.clear();
		for (auto &col : bf_plan->apply) {
			auto &col_bind = col;
			for (idx_t i = 0; i < bindings.size(); i++) {
				if (col_bind == bindings[i]) {
					bf_plan->bound_cols_apply.push_back(i);
					break;
				}
			}
		}
		if (bf_plan->bound_cols_apply.size() != bf_plan->apply.size()) {
			throw InternalException("Predicate Transfer: Failed to bind column reference.\n");
		}

		bindings = op.GetColumnBindings();
		VisitOperatorExpressions(op);
		return;
	}
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		// special case: comparison join
		auto &comp_join = op.Cast<LogicalComparisonJoin>();
		// first get the bindings of the LHS and resolve the LHS expressions
		VisitOperator(*comp_join.children[0]);
		for (auto &cond : comp_join.conditions) {
			VisitExpression(&cond.left);
		}
		// resolve any single-side predicates
		// for now, only ASOF supports this, and we are guaranteed that all right side predicates
		// have been pushed into a filter.
		if (comp_join.predicate) {
			D_ASSERT(op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN);
			VisitExpression(&comp_join.predicate);
		}
		// visit the duplicate eliminated columns on the LHS, if any
		for (auto &expr : comp_join.duplicate_eliminated_columns) {
			VisitExpression(&expr);
		}
		// then get the bindings of the RHS and resolve the RHS expressions
		VisitOperator(*comp_join.children[1]);
		for (auto &cond : comp_join.conditions) {
			VisitExpression(&cond.right);
		}
		// finally update the bindings with the result bindings of the join
		bindings = op.GetColumnBindings();
		return;
	}
	case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
		auto &comp_join = op.Cast<LogicalComparisonJoin>();
		// depending on whether the delim join has been flipped, get the appropriate bindings
		if (comp_join.delim_flipped) {
			VisitOperator(*comp_join.children[1]);
			for (auto &cond : comp_join.conditions) {
				VisitExpression(&cond.right);
			}
		} else {
			VisitOperator(*comp_join.children[0]);
			for (auto &cond : comp_join.conditions) {
				VisitExpression(&cond.left);
			}
		}
		// visit the duplicate eliminated columns
		for (auto &expr : comp_join.duplicate_eliminated_columns) {
			VisitExpression(&expr);
		}
		// now get the other side
		if (comp_join.delim_flipped) {
			VisitOperator(*comp_join.children[0]);
			for (auto &cond : comp_join.conditions) {
				VisitExpression(&cond.left);
			}
		} else {
			VisitOperator(*comp_join.children[1]);
			for (auto &cond : comp_join.conditions) {
				VisitExpression(&cond.right);
			}
		}
		// finally update the bindings with the result bindings of the join
		bindings = op.GetColumnBindings();
		return;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		// ANY join, this join is different because we evaluate the expression on the bindings of BOTH join sides at
		// once i.e. we set the bindings first to the bindings of the entire join, and then resolve the expressions of
		// this operator
		VisitOperatorChildren(op);
		bindings = op.GetColumnBindings();
		auto &any_join = op.Cast<LogicalAnyJoin>();
		if (any_join.join_type == JoinType::SEMI || any_join.join_type == JoinType::ANTI) {
			auto right_bindings = op.children[1]->GetColumnBindings();
			bindings.insert(bindings.end(), right_bindings.begin(), right_bindings.end());
		}
		if (any_join.join_type == JoinType::RIGHT_SEMI || any_join.join_type == JoinType::RIGHT_ANTI) {
			throw InternalException("RIGHT SEMI/ANTI any join not supported yet");
		}
		VisitOperatorExpressions(op);
		return;
	}
	case LogicalOperatorType::LOGICAL_CREATE_INDEX: {
		// CREATE INDEX statement, add the columns of the table with table index 0 to the binding set
		// afterwards bind the expressions of the CREATE INDEX statement
		auto &create_index = op.Cast<LogicalCreateIndex>();
		bindings = LogicalOperator::GenerateColumnBindings(0, create_index.table.GetColumns().LogicalColumnCount());
		VisitOperatorExpressions(op);
		return;
	}
	case LogicalOperatorType::LOGICAL_GET: {
		//! We first need to update the current set of bindings and then visit operator expressions
		bindings = op.GetColumnBindings();
		VisitOperatorExpressions(op);
		return;
	}
	case LogicalOperatorType::LOGICAL_INSERT: {
		//! We want to execute the normal path, but also add a dummy 'excluded' binding if there is a
		// ON CONFLICT DO UPDATE clause
		auto &insert_op = op.Cast<LogicalInsert>();
		if (insert_op.action_type != OnConflictAction::THROW) {
			// Get the bindings from the children
			VisitOperatorChildren(op);
			auto column_count = insert_op.table.GetColumns().PhysicalColumnCount();
			auto dummy_bindings = LogicalOperator::GenerateColumnBindings(insert_op.excluded_table_index, column_count);
			// Now insert our dummy bindings at the start of the bindings,
			// so the first 'column_count' indices of the chunk are reserved for our 'excluded' columns
			bindings.insert(bindings.begin(), dummy_bindings.begin(), dummy_bindings.end());
			if (insert_op.on_conflict_condition) {
				VisitExpression(&insert_op.on_conflict_condition);
			}
			if (insert_op.do_update_condition) {
				VisitExpression(&insert_op.do_update_condition);
			}
			VisitOperatorExpressions(op);
			bindings = op.GetColumnBindings();
			return;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR: {
		auto &ext_op = op.Cast<LogicalExtensionOperator>();
		ext_op.ResolveColumnBindings(*this, bindings);
		return;
	}
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
		auto &rec = op.Cast<LogicalRecursiveCTE>();
		VisitOperatorChildren(op);
		bindings = op.GetColumnBindings();

		for (auto &expr : rec.key_targets) {
			VisitExpression(&expr);
		}
		return;
	}
	default:
		break;
	}

	// general case
	// first visit the children of this operator
	VisitOperatorChildren(op);
	// now visit the expressions of this operator to resolve any bound column references
	VisitOperatorExpressions(op);
	// finally update the current set of bindings to the current set of column bindings
	bindings = op.GetColumnBindings();
}

unique_ptr<Expression> ColumnBindingResolver::VisitReplace(BoundColumnRefExpression &expr,
                                                           unique_ptr<Expression> *expr_ptr) {
	D_ASSERT(expr.depth == 0);
	// check the current set of column bindings to see which index corresponds to the column reference
	for (idx_t i = 0; i < bindings.size(); i++) {
		if (expr.binding == bindings[i]) {
			if (verify_only) {
				// in verification mode
				return nullptr;
			}
			return make_uniq<BoundReferenceExpression>(expr.GetAlias(), expr.return_type, i);
		}
	}
	// LCOV_EXCL_START
	// could not bind the column reference, this should never happen and indicates a bug in the code
	// generate an error message
	throw InternalException("Failed to bind column reference \"%s\" [%d.%d] (bindings: %s)", expr.GetAlias(),
	                        expr.binding.table_index, expr.binding.column_index,
	                        LogicalOperator::ColumnBindingsToString(bindings));
	// LCOV_EXCL_STOP
}

unordered_set<idx_t> ColumnBindingResolver::VerifyInternal(LogicalOperator &op) {
	unordered_set<idx_t> result;
	for (auto &child : op.children) {
		auto child_indexes = VerifyInternal(*child);
		for (auto index : child_indexes) {
			D_ASSERT(index != DConstants::INVALID_INDEX);
			if (result.find(index) != result.end()) {
				throw InternalException("Duplicate table index \"%lld\" found", index);
			}
			result.insert(index);
		}
	}
	auto indexes = op.GetTableIndex();
	for (auto index : indexes) {
		D_ASSERT(index != DConstants::INVALID_INDEX);
		if (result.find(index) != result.end()) {
			throw InternalException("Duplicate table index \"%lld\" found", index);
		}
		result.insert(index);
	}
	return result;
}

void ColumnBindingResolver::Verify(LogicalOperator &op) {
#ifdef DEBUG
	ColumnBindingResolver resolver(true);
	resolver.VisitOperator(op);
	VerifyInternal(op);
#endif
}

} // namespace duckdb
