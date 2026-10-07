"""
consensus_court_omics 适配层单元测试

覆盖范围（与用户规则"每次改动补单测"对应）：
1. build_candidate_evidence —— 五维评价 → 证据转换的真实/中性来源分级；
2. adjudicate_vaccine_candidates —— 真实 hold_court 集成：
   - 裁决返回结构完整；
   - ProjectEvent 落库（CONSENSUS_COURT_VERDICT）与案卷字段正确；
   - 空评价候选 → 不开庭、如实"无法裁决"；
   - ABP 定罪通道：异源对抗证据（存在/不存在）→ 定罪矛盾对
     intensity=3 + 升级条款 5 人工复核项在场。
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests.conftest import _create_all_tables
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# fixtures（复用 conftest 的全量建表逻辑，口径与 E2E 脚本一致）
# ---------------------------------------------------------------------------

@pytest.fixture()
def court_db():
    """内存 SQLite + 全量 ORM 表 + 真实项目/会话关联"""
    engine = create_engine(
        "sqlite:///:memory:", echo=False,
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    _create_all_tables(engine, skip_incompatible=True)
    Session = sessionmaker(bind=engine, autoflush=False)
    db = Session()

    from app.core.virtual_lab.models import AgentChatSession, ResearchProject

    project = ResearchProject(
        user_id="u-1", title="测试项目", goal="适配层单测",
        status="RUNNING", drug_type="vaccine",
        success_criteria={"ok": True},
    )
    db.add(project)
    db.flush()
    session = AgentChatSession(
        project_id=project.id, owner_user_id="u-1",
        title="unit-court", topic="单测庭审", mode="decision",
    )
    db.add(session)
    db.commit()
    yield db, project.id, session.id
    db.close()


def _evaluation_payload() -> dict:
    """构造与生产 evaluate_vaccine_candidate 输出同构的五维评价结果"""
    return {
        "composite_score": 0.4353,
        "dimensions": {
            "epitope_density": {
                "score": 0.3494,
                "raw": {"total": 893, "positive": 312},
                "note": "正得分表位 312/893（亲水/表面可及区段占比）",
            },
            "coverage": {
                "score": 0.62,
                "raw": {"coverage_ratio": 0.62},
                "note": "top 表位中心跨度/抗原长度",
            },
            "top_epitope_strength": {
                "score": 0.41,
                "raw": {"top_count": 5, "mean_score": 1.23},
                "note": "top5 表位均分 1.2300 / 归一化上限 3.0",
            },
            "conservation": {
                "score": 0.975,
                "raw": {"conservation_score": 0.975, "alignments": 5},
                "note": "真实序列比对保守性 0.9750（5 条参考序列）",
            },
            "structural_accessibility": {
                "score": 0.5,
                "raw": {},
                "note": "无结构证据，记中性值 0.5（不虚构）",
            },
        },
        "weights": {},
        "method": "parker_hydrophilicity_sliding_window",
        "data_completeness": 0.8,
    }


# ---------------------------------------------------------------------------
# 1. 证据构造
# ---------------------------------------------------------------------------

class TestBuildCandidateEvidence:
    """build_candidate_evidence 的维度→证据映射规则"""

    def test_one_evidence_per_dimension(self):
        """五个评价维度必须一一对应产出五条证据"""
        from app.core.virtual_lab.consensus_court_omics import (
            build_candidate_evidence,
        )

        evidences = build_candidate_evidence(
            candidate_id="cand-1", pathogen_name="犬腺病毒二型",
            antigen_name="hexon", evaluation=_evaluation_payload(),
        )
        assert len(evidences) == 5
        dims = {ev.metadata["dimension"] for ev in evidences}
        assert dims == set(_evaluation_payload()["dimensions"].keys())

    def test_real_vs_neutral_source_classification(self):
        """真实数据维度走高权源（experiment），中性/缺失走 agent_opinion"""
        from app.core.virtual_lab.consensus_court import EvidenceScorer
        from app.core.virtual_lab.consensus_court_omics import (
            build_candidate_evidence,
        )

        evidences = build_candidate_evidence(
            candidate_id="cand-1", pathogen_name="犬腺病毒二型",
            antigen_name="hexon", evaluation=_evaluation_payload(),
        )
        by_dim = {ev.metadata["dimension"]: ev for ev in evidences}
        assert by_dim["conservation"].source == "experiment"
        assert by_dim["structural_accessibility"].source == "agent_opinion"
        # 高权源权重 > 低权源权重（评分口径成立）
        w_real = EvidenceScorer.SOURCE_WEIGHTS["experiment"]
        w_neutral = EvidenceScorer.SOURCE_WEIGHTS["agent_opinion"]
        assert w_real > w_neutral

    def test_real_dimension_carries_reference_ids(self):
        """真实维度必须携带引用（否则会被 RuleEngine 初筛拒收）"""
        from app.core.virtual_lab.consensus_court_omics import (
            build_candidate_evidence,
        )

        evidences = build_candidate_evidence(
            candidate_id="cand-1", pathogen_name="犬腺病毒二型",
            antigen_name="hexon", evaluation=_evaluation_payload(),
        )
        by_dim = {ev.metadata["dimension"]: ev for ev in evidences}
        assert by_dim["conservation"].reference_ids
        # 中性维度不携带引用（诚实反映"无实测出处"）
        assert not by_dim["structural_accessibility"].reference_ids


# ---------------------------------------------------------------------------
# 2. 真实开庭（hold_court 集成 + 事件落库）
# ---------------------------------------------------------------------------

class TestAdjudicateVaccineCandidates:
    """adjudicate_vaccine_candidates 的真实 hold_court 集成语义"""

    def test_positive_candidate_no_false_conviction(self, court_db):
        """ABP 核心主张：真实无矛盾评价数据不得被定罪（无罪推定）"""
        from app.core.virtual_lab.consensus_court_omics import (
            adjudicate_vaccine_candidates,
        )

        db, project_id, session_id = court_db
        results = adjudicate_vaccine_candidates(
            db=db, session_id=session_id, project_id=project_id,
            candidates=[{
                "candidate_id": "cand-1",
                "pathogen_name": "犬腺病毒二型（CAV-2）",
                "antigen_name": "hexon",
                "evaluation": _evaluation_payload(),
            }],
        )
        assert len(results) == 1
        verdict = results[0]
        # 真实评价集不含矛盾对：不得出现定罪强度
        assert verdict["max_contradiction_intensity"] == 0
        # 裁决为三值之一（支持/有条件支持/驳回），不得"无法裁决"
        assert verdict["decision"] in ("支持提案", "有条件支持", "驳回提案")
        assert verdict["evidence_count"] == 5

    def test_verdict_persisted_as_project_event(self, court_db):
        """裁决必须以 CONSENSUS_COURT_VERDICT 事件落库且案卷完整"""
        from app.core.virtual_lab.consensus_court import (
            COURT_VERDICT_EVENT_TYPE,
        )
        from app.core.virtual_lab.consensus_court_omics import (
            adjudicate_vaccine_candidates,
        )
        from app.core.virtual_lab.models import ProjectEvent

        db, project_id, session_id = court_db
        adjudicate_vaccine_candidates(
            db=db, session_id=session_id, project_id=project_id,
            candidates=[{
                "candidate_id": "cand-1",
                "pathogen_name": "犬腺病毒二型（CAV-2）",
                "antigen_name": "hexon",
                "evaluation": _evaluation_payload(),
            }],
        )
        events = (
            db.query(ProjectEvent)
            .filter(
                ProjectEvent.project_id == project_id,
                ProjectEvent.event_type == COURT_VERDICT_EVENT_TYPE,
            )
            .all()
        )
        assert len(events) == 1
        payload = events[0].payload
        # 案卷关键字段（可审计性契约）
        # correlation_id 是 ProjectEvent 表列（非 payload 键），
        # _persist_verdict_event 用 session_id 填充，供 L3 会话聚合
        assert events[0].correlation_id == session_id
        assert payload["session_id"] == session_id
        assert "topic" in payload and "hexon" in payload["topic"]
        assert len(payload["evidence_scores"]) >= 1
        assert payload["evidence_fingerprints"]

    def test_empty_evaluation_not_adjudicated(self, court_db):
        """无评价数据不得开庭：如实'无法裁决'并标记升级（诚实降级）"""
        from app.core.virtual_lab.consensus_court_omics import (
            adjudicate_vaccine_candidates,
        )

        db, project_id, session_id = court_db
        results = adjudicate_vaccine_candidates(
            db=db, session_id=session_id, project_id=project_id,
            candidates=[{
                "candidate_id": "cand-empty",
                "pathogen_name": "未评病原",
                "antigen_name": None,
                "evaluation": {},
            }],
        )
        assert results[0]["decision"] == "无法裁决"
        assert results[0]["escalated"] is True
        assert results[0]["evidence_count"] == 0

    def test_adversarial_contradiction_convicted(self, court_db):
        """异源对抗证据（存在/不存在）→ ABP 定罪 + 升级条款 5 复核项"""
        from app.core.virtual_lab.consensus_court import ConsensusCourt, Evidence
        from app.core.virtual_lab.consensus_court_omics import (
            build_trial_topic,
        )

        db, project_id, session_id = court_db
        md = {"target": "hexon", "molecule": "hexon"}
        ev_a = Evidence(
            evidence_id="adv:presence", source="pubmed",
            content="犬腺病毒二型（CAV-2） 疫苗靶点 hexon：基因组序列存在该抗原编码基因（比对保守性 0.9750），蛋白序列完整可用。",
            role="molecular-detectologist", timestamp=datetime.utcnow(),
            reference_ids=["r1"], metadata=md,
        )
        ev_b = Evidence(
            evidence_id="adv:absence", source="uniprot",
            content="犬腺病毒二型（CAV-2） 疫苗靶点 hexon：独立核查结论——该抗原不存在于参考基因组，无保守性证据。",
            role="structural-biologist", timestamp=datetime.utcnow(),
            reference_ids=["r2"], metadata=md,
        )
        verdict = ConsensusCourt(db).hold_court(
            session_id=session_id,
            topic=build_trial_topic("犬腺病毒二型（CAV-2）", "hexon"),
            evidence_list=[ev_a, ev_b],
        )
        # 定罪：对抗级（intensity=3）矛盾对
        assert verdict.max_contradiction_intensity == 3
        # 升级条款 5：对抗级强制人工复核项在场
        assert any("升级条款 5" in item for item in verdict.human_override_items)
        assert verdict.escalated is True


# ---------------------------------------------------------------------------
# 4. 措辞协议锁（架构审查 MAJOR-1/MAJOR-3 修复配套）
# ---------------------------------------------------------------------------

class TestWordingProtocolLock:
    """上游真实评价输出 → 适配层来源分级的协议锁

    守护目标：_dimension_source 与 vaccine_evaluation 诚实降级措辞
    之间的跨模块隐式协议。上游改写 note 文案（如"数据不足："替代
    "insufficient_data:"）时，本组测试必须在 CI 立即失败，而不是
    让无数据证据静默升格为 experiment(0.75) 高权源。零网络、零 IO。
    """

    def test_all_insufficient_from_real_upstream(self):
        """上游全缺失路径：五维证据必须全部低权 agent_opinion 且无引用"""
        from app.core.virtual_lab.vaccine_evaluation import (
            evaluate_vaccine_candidate,
        )
        from app.core.virtual_lab.consensus_court_omics import (
            build_candidate_evidence,
        )

        evaluation = evaluate_vaccine_candidate(
            epitopes_payload={}, pathogen_name="未知病原X-协议锁",
        )
        evidences = build_candidate_evidence(
            candidate_id="lock-1", pathogen_name="未知病原X-协议锁",
            antigen_name="unknown-antigen-xyz", evaluation=evaluation,
        )
        assert len(evidences) == 5
        for ev in evidences:
            assert ev.source == "agent_opinion", (
                f"维度 {ev.metadata['dimension']} note={ev.metadata.get('score')} "
                "上游无数据维度被升格为高权源——诚实降级措辞协议已漂移"
            )
            assert ev.reference_ids == [], (
                "无数据维度不应携带引用（引用只能来自真实计算实据）"
            )

    def test_real_evidence_from_real_upstream(self):
        """上游真实保守性/结构证据：对应维度必须升为 experiment 且引用可溯源"""
        from app.core.virtual_lab.vaccine_evaluation import (
            evaluate_vaccine_candidate,
        )
        from app.core.virtual_lab.consensus_court_omics import (
            build_candidate_evidence,
        )

        evaluation = evaluate_vaccine_candidate(
            epitopes_payload={},
            pathogen_name="协议锁病原",
            antigen_name="hexon",
            conservation_evidence={"conservation_score": 0.97, "alignments": 5},
            structure_evidence={"surface_residue_ratio": 0.62, "pdb_id": "6KU9"},
            method="parker_hydrophilicity_sliding_window",
        )
        evidences = {
            ev.metadata["dimension"]: ev
            for ev in build_candidate_evidence(
                candidate_id="lock-2", pathogen_name="协议锁病原",
                antigen_name="hexon", evaluation=evaluation,
            )
        }
        cons = evidences["conservation"]
        assert cons.source == "experiment"
        # 真实溯源（MAJOR-2）：引用串携带方法名与实据键，非自指裸串
        assert any("method=parker_hydrophilicity_sliding_window" in ref
                   for ref in cons.reference_ids)
        assert any("alignments=5" in ref for ref in cons.reference_ids)
        acc = evidences["structural_accessibility"]
        assert acc.source == "experiment"
        assert any("pdb_id=6KU9" in ref for ref in acc.reference_ids)
        # 表位维度无输入 → 仍须低权（不得因同批其他维度真实而搭车升权）
        assert evidences["epitope_density"].source == "agent_opinion"


# ---------------------------------------------------------------------------
# 5. 归属校验与批次容错（架构审查 BLOCKER-1 / MAJOR-4 修复）
# ---------------------------------------------------------------------------

class TestOwnershipAndResilience:
    """adjudicate_vaccine_candidates 的 fail-fast 校验与批次容错"""

    def test_wrong_project_id_rejected(self, court_db):
        """会话归属项目与传入 project_id 不一致 → ValueError（防越权写入）"""
        from app.core.virtual_lab.consensus_court_omics import (
            adjudicate_vaccine_candidates,
        )

        db, _project_id, session_id = court_db
        with pytest.raises(ValueError, match="不一致"):
            adjudicate_vaccine_candidates(
                db=db, session_id=session_id, project_id="other-project",
                candidates=[{
                    "candidate_id": "c-1", "pathogen_name": "P",
                    "antigen_name": "A", "evaluation": _evaluation_payload(),
                }],
            )

    def test_missing_session_rejected(self, court_db):
        """会话不存在 → ValueError（案卷无处归档，不落半成品）"""
        from app.core.virtual_lab.consensus_court_omics import (
            adjudicate_vaccine_candidates,
        )

        db, project_id, _session_id = court_db
        with pytest.raises(ValueError, match="不存在"):
            adjudicate_vaccine_candidates(
                db=db, session_id="no-such-session", project_id=project_id,
                candidates=[{
                    "candidate_id": "c-1", "pathogen_name": "P",
                    "antigen_name": "A", "evaluation": _evaluation_payload(),
                }],
            )

    def test_blank_candidate_id_rejected(self, court_db):
        """candidate_id 为空 → ValueError（证据 ID 会退化成 ':dim'）"""
        from app.core.virtual_lab.consensus_court_omics import (
            adjudicate_vaccine_candidates,
        )

        db, project_id, session_id = court_db
        with pytest.raises(ValueError, match="candidate_id"):
            adjudicate_vaccine_candidates(
                db=db, session_id=session_id, project_id=project_id,
                candidates=[{
                    "candidate_id": "  ", "pathogen_name": "P",
                    "antigen_name": "A", "evaluation": _evaluation_payload(),
                }],
            )

    def test_kernel_exception_does_not_break_batch(
        self, court_db, monkeypatch,
    ):
        """单候选内核异常：如实产出'无法裁决+error'，其余候选照常裁决"""
        from app.core.virtual_lab import consensus_court_omics as omics

        db, project_id, session_id = court_db
        original = omics.ConsensusCourt.hold_court

        def flaky_hold_court(self, session_id, topic, evidence_list):
            # 模拟内核在特定候选上崩溃（如畸形评价引发的类型错误）
            if "坏候选" in topic:
                raise RuntimeError("模拟内核崩溃")
            return original(self, session_id, topic, evidence_list)

        monkeypatch.setattr(
            omics.ConsensusCourt, "hold_court", flaky_hold_court,
        )
        results = omics.adjudicate_vaccine_candidates(
            db=db, session_id=session_id, project_id=project_id,
            candidates=[
                {"candidate_id": "bad-1", "pathogen_name": "坏候选病原",
                 "antigen_name": "hexon", "evaluation": _evaluation_payload()},
                {"candidate_id": "good-1", "pathogen_name": "犬腺病毒二型",
                 "antigen_name": "hexon", "evaluation": _evaluation_payload()},
            ],
        )
        bad, good = results
        assert bad["decision"] == "无法裁决"
        assert bad["escalated"] is True
        assert "RuntimeError" in bad["error"]
        # 批次不中断：第二个候选照常完成真实开庭
        assert good["decision"] != "无法裁决"
        assert good["max_contradiction_intensity"] == 0

    def test_malformed_dimension_value_skipped(self):
        """维度值非 dict（上游 JSON 列脏数据）：跳过该维度而非崩溃"""
        from app.core.virtual_lab.consensus_court_omics import (
            build_candidate_evidence,
        )

        evaluation = _evaluation_payload()
        evaluation["dimensions"]["coverage"] = "not-a-dict"
        evidences = build_candidate_evidence(
            candidate_id="cand-dirty", pathogen_name="P",
            antigen_name="A", evaluation=evaluation,
        )
        dims = {ev.metadata["dimension"] for ev in evidences}
        assert "coverage" not in dims
        assert len(evidences) == 4


# ---------------------------------------------------------------------------
# 6. 案卷去重（架构审查 MINOR-4 修复）
# ---------------------------------------------------------------------------

class TestVerdictDeduplication:
    """同一评价数据重复仲裁：案卷如实落库（审计不可变）但标注 duplicate_of"""

    def test_duplicate_adjudication_annotated(self, court_db):
        """同候选二次裁决 → duplicate_of 指向首案 event_id"""
        from app.core.virtual_lab.consensus_court import (
            COURT_VERDICT_EVENT_TYPE,
        )
        from app.core.virtual_lab.consensus_court_omics import (
            adjudicate_vaccine_candidates,
        )
        from app.core.virtual_lab.models import ProjectEvent

        db, project_id, session_id = court_db
        candidate = [{
            "candidate_id": "cand-dup", "pathogen_name": "犬腺病毒二型",
            "antigen_name": "hexon", "evaluation": _evaluation_payload(),
        }]
        first = adjudicate_vaccine_candidates(
            db=db, session_id=session_id, project_id=project_id,
            candidates=candidate,
        )
        # 首案无重复标注
        assert "duplicate_of" not in first[0]
        # 同一评价数据二次仲裁：确定性指纹一致 → 标注 duplicate_of
        second = adjudicate_vaccine_candidates(
            db=db, session_id=session_id, project_id=project_id,
            candidates=candidate,
        )
        events = (
            db.query(ProjectEvent)
            .filter(ProjectEvent.event_type == COURT_VERDICT_EVENT_TYPE)
            .order_by(ProjectEvent.created_at.asc())
            .all()
        )
        # 案卷不可变：两份都如实落库
        assert len(events) == 2
        assert second[0]["duplicate_of"] == events[0].id

    def test_different_candidate_not_marked_duplicate(self, court_db):
        """不同候选（evidence_id 不同 → 指纹不同）不得误判为重复"""
        from app.core.virtual_lab.consensus_court_omics import (
            adjudicate_vaccine_candidates,
        )

        db, project_id, session_id = court_db
        results = adjudicate_vaccine_candidates(
            db=db, session_id=session_id, project_id=project_id,
            candidates=[
                {"candidate_id": "cand-a", "pathogen_name": "犬腺病毒二型",
                 "antigen_name": "hexon", "evaluation": _evaluation_payload()},
                {"candidate_id": "cand-b", "pathogen_name": "犬腺病毒二型",
                 "antigen_name": "hexon", "evaluation": _evaluation_payload()},
            ],
        )
        assert "duplicate_of" not in results[0]
        assert "duplicate_of" not in results[1]


# ---------------------------------------------------------------------------
# 7. 时间戳时区口径与 source/role 归一（架构审查 INFO-2/INFO-3）
# ---------------------------------------------------------------------------

class TestTimestampAndNormalization:
    """aware-UTC 证据时间戳在内核评分链路的正确性 + 归一口径"""

    def test_aware_timestamp_recency_not_degraded(self):
        """刚生成的 aware-UTC 证据：时效分不得因 aware-naive 相减
        TypeError 被吞而静默退化为 0.5（应接近满分衰减起点 1.0）"""
        from app.core.virtual_lab.consensus_court import (
            EvidenceScorer,
        )
        from app.core.virtual_lab.consensus_court_omics import (
            _utc_now,
            build_candidate_evidence,
        )

        evidences = build_candidate_evidence(
            candidate_id="cand-tz", pathogen_name="犬腺病毒二型",
            antigen_name="hexon", evaluation=_evaluation_payload(),
            # 显式传 aware-UTC 时间戳，验证内核 _recency_score
            evidence_time=_utc_now(),
        )
        scorer = EvidenceScorer()
        topic = "犬腺病毒二型 疫苗抗原 hexon 靶点候选证据"
        for ev in evidences:
            score = scorer.score(ev, topic)
            # 新证据时效分应在 (0.5, 1.0]：若 aware-naive 相减抛错
            # 会被 except 吞掉并恒返 0.5
            assert score.recency_score > 0.5, (
                f"aware-UTC 时间戳时效分退化为 {score.recency_score}，"
                "_recency_score 时区兼容已失效"
            )

    def test_naive_legacy_timestamp_still_works(self):
        """历史存量 naive-UTC 时间戳：时效分正常计算（向后兼容）"""
        from app.core.virtual_lab.consensus_court import EvidenceScorer

        scorer = EvidenceScorer()
        # naive UTC 30 天前（半衰期口径下应有确定衰减，非退化值）
        legacy = datetime.utcnow() - timedelta(days=30)
        recency = scorer._recency_score(legacy)
        assert 0.0 < recency < 1.0
        assert recency != 0.5

    def test_source_and_role_whitespace_case_normalized(self):
        """source/role 带大小写与首尾空白：归一后权重命中不变"""
        from app.core.virtual_lab.consensus_court import (
            Evidence,
            EvidenceScorer,
            PIAgentArbiter,
        )
        from app.core.virtual_lab.consensus_court_omics import _DIMENSION_ROLE

        ev = Evidence(
            evidence_id="norm-1", source=" Experiment ",
            content="疫苗候选维度证据内容足够长度以便通过相关性计算",
            role=f"  {_DIMENSION_ROLE.upper()} ",
            timestamp=datetime.utcnow(),
            reference_ids=["r1"],
        )
        score = EvidenceScorer().score(ev, "疫苗 抗原 表位")
        # " Experiment " 归一后必须命中 experiment(0.75) 而非兜底 0.2
        assert score.source_score == 0.75
        # role 归一命中 drug-researcher(0.9)：验证 PI 权重表查询口径
        # （与 PIAgentArbiter.arbitrate 内 strip().lower() 同口径）
        assert PIAgentArbiter.ROLE_WEIGHTS.get(
            ev.role.strip().lower()
        ) == 0.9
