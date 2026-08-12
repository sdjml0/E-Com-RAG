import math
import logging
from typing import List, Dict, Any, Tuple
from app.config import settings
from app.schemas import ScoreBreakdown

logger = logging.getLogger("rrf_engine")

class RRFRanker:
    """Reciprocal Rank Fusion (RRF) & Dynamic Price Penalty Engine."""

    def __init__(
        self,
        k: float = settings.RRF_K,
        default_weight_text: float = settings.WEIGHT_TEXT,
        default_weight_image: float = settings.WEIGHT_IMAGE,
        default_weight_bm25: float = settings.WEIGHT_BM25,
        alpha: float = settings.PRICE_PENALTY_ALPHA,
        min_factor: float = settings.MIN_PRICE_PENALTY_FACTOR
    ):
        self.k = k
        self.default_w_text = default_weight_text
        self.default_w_image = default_weight_image
        self.default_w_bm25 = default_weight_bm25
        self.alpha = alpha
        self.min_factor = min_factor

    def compute_rrf_and_penalties(
        self,
        text_hits: List[Any],
        image_hits: List[Any],
        target_price: float | None = None,
        weight_text: float | None = None,
        weight_image: float | None = None
    ) -> List[Tuple[str, float, ScoreBreakdown, dict]]:
        """
        Fuses ranked candidate lists from text and vision vector queries using RRF,
        then applies price elasticity soft dampening penalties.
        """
        w_text = weight_text if weight_text is not None else self.default_w_text
        w_image = weight_image if weight_image is not None else self.default_w_image
        
        # Maps product_id -> {text_rank, visual_rank, payload}
        product_ranks: Dict[str, Dict[str, Any]] = {}
        payload_map: Dict[str, dict] = {}

        # 1. Process Text Vector Search Hits
        for rank_idx, hit in enumerate(text_hits):
            payload = hit.payload
            pid = payload["product_id"]
            payload_map[pid] = payload
            if pid not in product_ranks:
                product_ranks[pid] = {"text_rank": None, "visual_rank": None}
            product_ranks[pid]["text_rank"] = rank_idx + 1

        # 2. Process Image Vector Search Hits
        for rank_idx, hit in enumerate(image_hits):
            payload = hit.payload
            pid = payload["product_id"]
            payload_map[pid] = payload
            if pid not in product_ranks:
                product_ranks[pid] = {"text_rank": None, "visual_rank": None}
            product_ranks[pid]["visual_rank"] = rank_idx + 1

        # 3. Calculate RRF Fusion Score
        scored_results = []
        for pid, ranks in product_ranks.items():
            t_rank = ranks["text_rank"]
            v_rank = ranks["visual_rank"]

            rrf_score = 0.0
            if t_rank is not None:
                rrf_score += w_text / (self.k + t_rank)
            if v_rank is not None:
                rrf_score += w_image / (self.k + v_rank)

            # 4. Compute Price Soft Elasticity Penalty if target price specified
            price_penalty = 1.0
            item_price = payload_map[pid].get("price", 0.0)
            if target_price and target_price > 0 and item_price > 0:
                diff_ratio = abs(item_price - target_price) / target_price
                price_penalty = max(self.min_factor, 1.0 - (self.alpha * diff_ratio))

            final_score = rrf_score * price_penalty

            breakdown = ScoreBreakdown(
                text_rank=t_rank,
                visual_rank=v_rank,
                rrf_score=round(rrf_score, 6),
                price_penalty=round(price_penalty, 4)
            )

            scored_results.append((pid, final_score, breakdown, payload_map[pid]))

        # Sort descending by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results

rrf_ranker = RRFRanker()
