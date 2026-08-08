from sqlmodel import Session, select

from app.core.config import settings
from app.models.entities import Strategy

_collection = None
_embed_func = None


def _embedder_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import chromadb  # noqa: F401

        return True
    except ImportError:
        return False


def _get_embedder():
    global _embed_func
    if _embed_func is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embeddings_model)
        _embed_func = lambda texts: model.encode(texts, normalize_embeddings=True).tolist()  # noqa: E731
    return _embed_func


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_dir)
        _collection = client.get_or_create_collection("strategies")
    return _collection


class RagService:
    """Strategy RAG layer. Falls back to keyword search when the optional
    embedding/vector deps are not installed."""

    def __init__(self, session: Session):
        self.session = session

    @property
    def available(self) -> bool:
        return _embedder_available()

    def index_strategy(self, strategy: Strategy) -> None:
        if not self.available:
            strategy.embedding_id = strategy.id
            self.session.commit()
            return
        embed = _get_embedder()
        collection = _get_collection()
        text = f"{strategy.name}\n{strategy.body_markdown}"
        if getattr(strategy, "concept_markdown", ""):
            text = f"{strategy.name}\n{strategy.concept_markdown}\n{strategy.body_markdown}"
        vector = embed([text])[0]
        collection.upsert(
            ids=[strategy.id],
            documents=[text],
            metadatas=[{
                "name": strategy.name,
                "asset_class": strategy.asset_class,
                "timeframe": strategy.timeframe,
                "status": strategy.status,
            }],
            embeddings=[vector],
        )
        strategy.embedding_id = strategy.id
        self.session.commit()

    def search(self, query: str, top_k: int = 3) -> list[str]:
        if not self.available:
            return self._fallback(query, top_k)
        embed = _get_embedder()
        collection = _get_collection()
        vector = embed([query])[0]
        res = collection.query(query_embeddings=[vector], n_results=top_k)
        return res.get("documents", [[]])[0]

    def search_active(self, query: str, top_k: int = 3,
                      asset_class: str | None = None) -> list[str]:
        """Retrieve only APPROVED strategies, optionally narrowed to an asset class.

        Scanning must ground itself in the rules the trader actually approved; draft
        or archived bodies should never leak into live setup-detection context.
        """
        if self.available:
            embed = _get_embedder()
            collection = _get_collection()
            vector = embed([query])[0]
            where: dict = {"status": "active"}
            if asset_class:
                where["asset_class"] = _normalize_class(asset_class)
            try:
                res = collection.query(
                    query_embeddings=[vector], n_results=top_k, where=where,
                )
                docs = res.get("documents", [[]])[0]
                if docs:
                    return docs
            except Exception:  # noqa: BLE001
                # Metadata filter may not match (embedding not refreshed); fall through.
                pass
        return self._fallback(query, top_k, active_only=True, asset_class=asset_class)

    def search_concepts(self, query: str, top_k: int = 6,
                        asset_class: str | None = None) -> list[str]:
        """Retrieve the INGESTED CONCEPT LIBRARY for synthesis grounding.

        Unlike :meth:`search_active` (scanner path — approved rules only), the
        Synthesizer must ground on the user's saved REFERENCE concepts PLUS the
        approved strategies, so this searches both ``reference`` and ``active``
        docs. Returns the indexed text (name + concept + body), never empty
        strings, so concept-only references are actually retrievable.
        """
        if self.available:
            embed = _get_embedder()
            collection = _get_collection()
            vector = embed([query])[0]
            where: dict = {"status": {"$in": ["reference", "active"]}}
            if asset_class:
                where["asset_class"] = _normalize_class(asset_class)
            try:
                res = collection.query(
                    query_embeddings=[vector], n_results=top_k, where=where,
                )
                docs = res.get("documents", [[]])[0]
                if docs:
                    return [d for d in docs if isinstance(d, str) and d.strip()]
            except Exception:  # noqa: BLE001
                # Metadata filter mismatch (embedding not refreshed); fall through.
                pass
        return self._fallback(
            query, top_k,
            statuses=("reference", "active"),
            asset_class=asset_class,
        )

    def delete_strategy(self, strategy_id: str) -> None:
        if not self.available:
            return
        try:
            _get_collection().delete(ids=[strategy_id])
        except Exception:
            pass

    def _fallback(self, query: str, top_k: int,
                  active_only: bool = False,
                  statuses: tuple[str, ...] | None = None,
                  asset_class: str | None = None) -> list[str]:
        stmt = select(Strategy)
        if active_only:
            stmt = stmt.where(Strategy.status == "active")
        if statuses:
            stmt = stmt.where(Strategy.status.in_(statuses))
        if asset_class:
            stmt = stmt.where(Strategy.asset_class == _normalize_class(asset_class))
        rows = self.session.exec(stmt.limit(50)).all()
        scored = []
        query_tokens = {t for t in _tokens(query)}
        for s in rows:
            # Indexed-style text (same as index_strategy) so concept-only
            # references are retrievable, not empty bodies.
            text = f"{s.name}\n{s.concept_markdown or ''}\n{s.body_markdown}"
            hay = _tokens(text)
            overlap = hay & query_tokens
            score = len(overlap) / max(len(query_tokens), 1) if overlap else 0.0
            scored.append((score, text.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [md for _, md in scored[:top_k] if md]


def _normalize_class(asset_class: str | None) -> str:
    c = (asset_class or "").strip().lower()
    aliases = {
        "forex": "forex", "fx": "forex", "currency": "forex",
        "crypto": "cripto", "cripto": "cripto", "cryptocurrency": "cripto",
        "commodities": "commodities", "commodity": "commodities", "metals": "commodities",
        "indices": "indices", "index": "indices", "stocks": "stocks", "equities": "stocks",
    }
    return aliases.get(c, c)


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}