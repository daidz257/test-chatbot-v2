import unittest
from rag_demo import db, quota, evaluation, workflow, advanced_rag, agentic_patterns

class RegressionTests(unittest.TestCase):
    def test_db_initialization(self):
        db.init_db()
        kbs = db.list_knowledge_bases()
        self.assertGreater(len(kbs), 0)
        scenarios = db.list_scenarios()
        self.assertGreater(len(scenarios), 0)

    def test_evaluation_metrics_calculation(self):
        case = {
            "id": "test-case-1",
            "question": "Ứng viên có bằng cấp gì?",
            "reference_answer": "Bằng Cử nhân CNTT.",
            "expected_sources": ["CV.pdf"],
            "answerable": True,
        }
        sources = [
            {"ref": "S1", "file_name": "CV.pdf", "page_number": 1, "chunk_index": 0}
        ]
        metrics = evaluation.case_metrics(case, "Bằng Cử nhân CNTT [S1].", sources)
        self.assertTrue(metrics["retrieval_hit"])
        self.assertEqual(metrics["expected_source_recall"], 1.0)
        self.assertTrue(metrics["citation_check_passed"])
        self.assertFalse(metrics["abstained"])

        unanswerable_case = {
            "id": "test-unanswerable",
            "question": "Điểm IELTS?",
            "reference_answer": "Chưa đủ căn cứ.",
            "expected_sources": ["CV.pdf"],
            "answerable": False,
        }
        metrics_abstain = evaluation.case_metrics(unanswerable_case, "Chưa đủ căn cứ để trả lời.", sources)
        self.assertTrue(metrics_abstain["abstained"])
        self.assertTrue(metrics_abstain["abstention_correct"])

    def test_workflow_state_graph_compilation_and_execution(self):
        def dummy_node(state):
            return {}

        state_baseline = workflow.RAGState(
            needs_plan=False,
            needs_context_grade=False,
            needs_audit=False,
            needs_reflection=False,
            query="test baseline",
            history=[],
        )
        res_baseline = workflow.run_rag_graph(
            state_baseline,
            plan_node=dummy_node,
            retrieve_node=dummy_node,
            grade_context_node=dummy_node,
            retry_query_node=dummy_node,
            compress_node=dummy_node,
            answer_node=dummy_node,
            audit_node=dummy_node,
            revise_answer_node=dummy_node,
            abstain_node=dummy_node,
        )
        self.assertIsNotNone(res_baseline)

        state_crag = workflow.RAGState(
            needs_plan=False,
            needs_context_grade=True,
            context_grade={"sufficient": True},
            needs_audit=False,
            needs_reflection=False,
            query="test crag",
            history=[],
        )
        res_crag = workflow.run_rag_graph(
            state_crag,
            plan_node=dummy_node,
            retrieve_node=dummy_node,
            grade_context_node=dummy_node,
            retry_query_node=dummy_node,
            compress_node=dummy_node,
            answer_node=dummy_node,
            audit_node=dummy_node,
            revise_answer_node=dummy_node,
            abstain_node=dummy_node,
        )
        self.assertIsNotNone(res_crag)

        state_reflection = workflow.RAGState(
            needs_plan=False,
            needs_context_grade=False,
            needs_audit=True,
            needs_reflection=True,
            reflection_count=0,
            audit={"llm_audit": {"grade": 2, "unsupported_claims": ["claim 1"]}},
            query="test reflection",
            history=[],
        )
        self.assertTrue(workflow.audit_needs_revision(state_reflection["audit"]))

    def test_quota_request_meter_token_estimation(self):
        meter = quota.RequestMeter()
        meter.record_attempt("generation", "test-label")
        data = meter.as_dict()
        self.assertEqual(data["api_calls"], 1)
        self.assertEqual(data["generation_calls"], 1)
        self.assertIn("test-label", data["labels"])

    def test_agentic_function_calling_route_options(self):
        self.assertIn("document_count", agentic_patterns.ROUTE_OPTIONS)
        self.assertIn("document_list", agentic_patterns.ROUTE_OPTIONS)
        self.assertIn("all_documents", agentic_patterns.ROUTE_OPTIONS)
        self.assertIn("focused_rag", agentic_patterns.ROUTE_OPTIONS)

if __name__ == "__main__":
    unittest.main()

