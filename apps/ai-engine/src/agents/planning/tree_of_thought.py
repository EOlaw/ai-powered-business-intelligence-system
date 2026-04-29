"""
InsightSerenity AI Engine — Tree of Thought Planning
=====================================================
Tree of Thoughts (ToT) is a generalisation of Chain-of-Thought that
explores multiple reasoning paths simultaneously, evaluates them, and
selects the most promising one to continue.

Reference: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
           Yao et al., 2023.

Key idea:
    Instead of generating one linear chain of thought, ToT generates k
    candidate "thoughts" at each step, evaluates each one with a value
    function, and explores the best branches — like a tree search.

    Root: [problem]
    Level 1: [thought_1, thought_2, thought_3]
    Level 2: [thought_1.1, thought_1.2, thought_2.1, ...] (best branches expanded)
    ...
    Leaf: [final_answer]

Search strategies:
    BFS (breadth-first): Expand all nodes at the same depth before going deeper.
                         Good for problems where the right answer requires
                         exploring many parallel paths.
    DFS (depth-first):   Go deep on the most promising branch first.
                         Good for problems with clear progress indicators.

Evaluation:
    After generating candidate thoughts, each is scored by asking the LLM:
    "Given the problem and this reasoning step, is this a good path?
     Rate from 1 (very bad) to 10 (very promising)."

When to use ToT vs ReAct:
    ReAct: Most tasks. Sequential, deterministic, single path. Fast.
    ToT:   Complex tasks where the first approach often fails:
           - Multi-step math with many sub-problems
           - Creative writing with hard constraints
           - Planning with many valid alternatives
           - Puzzles that require backtracking
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ThoughtNode:
    """A single node in the thought tree."""
    thought:      str
    depth:        int
    score:        float           = 0.0
    children:     List["ThoughtNode"] = field(default_factory=list)
    parent:       Optional["ThoughtNode"] = None
    is_terminal:  bool            = False
    answer:       Optional[str]   = None

    def path_to_root(self) -> List[str]:
        """Return all thoughts from root to this node."""
        path  = []
        node  = self
        while node is not None:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))


class TreeOfThoughtPlanner:
    """
    Tree of Thoughts planner for complex multi-step reasoning.

    Generates k candidate thoughts at each step, scores them with the
    LLM, and expands the best-scoring branches.

    Args:
        generator:   TextGenerator (our LLM).
        n_thoughts:  Number of candidate thoughts per node. Default 3.
        max_depth:   Maximum tree depth (reasoning steps). Default 4.
        n_best:      Number of best nodes to expand at each level (BFS). Default 2.
        search:      "bfs" or "dfs". Default "bfs".
        max_tokens:  Max tokens per thought generation.
    """

    def __init__(
        self,
        generator:  Any,
        n_thoughts: int  = 3,
        max_depth:  int  = 4,
        n_best:     int  = 2,
        search:     str  = "bfs",
        max_tokens: int  = 200,
    ) -> None:
        self.generator  = generator
        self.n_thoughts = n_thoughts
        self.max_depth  = max_depth
        self.n_best     = n_best
        self.search     = search
        self.max_tokens = max_tokens

    def solve(self, problem: str) -> Tuple[str, List[str]]:
        """
        Solve a problem using Tree of Thoughts search.

        Args:
            problem: The problem or question to solve.

        Returns:
            Tuple (final_answer, reasoning_path).
            reasoning_path is the list of thoughts from root to the final answer.
        """
        logger.info("ToT solving", problem=problem[:80], strategy=self.search)

        # Create root node
        root = ThoughtNode(thought=f"Problem: {problem}", depth=0)

        if self.search == "bfs":
            return self._bfs_solve(problem, root)
        else:
            return self._dfs_solve(problem, root)

    def _bfs_solve(self, problem: str, root: ThoughtNode) -> Tuple[str, List[str]]:
        """
        Breadth-first search through the thought tree.

        At each depth level: generate thoughts → evaluate → keep best n_best.
        """
        current_level = [root]

        for depth in range(1, self.max_depth + 1):
            next_level = []

            for node in current_level:
                # Generate k candidate thoughts from this node
                candidates = self._generate_thoughts(problem, node)

                # Evaluate each candidate
                for thought in candidates:
                    child = ThoughtNode(
                        thought=thought,
                        depth=depth,
                        parent=node,
                    )
                    child.score = self._evaluate_thought(problem, child)

                    # Check if this thought is a final answer
                    if self._is_terminal(thought):
                        answer = self._extract_answer(thought)
                        path   = child.path_to_root()
                        logger.info("ToT found answer at depth", depth=depth)
                        return answer, path

                    node.children.append(child)
                    next_level.append(child)

            # Keep only the best n_best nodes to expand
            next_level.sort(key=lambda n: n.score, reverse=True)
            current_level = next_level[:self.n_best]

            if not current_level:
                break

        # No terminal node found — use the highest-scoring leaf
        if current_level:
            best = max(current_level, key=lambda n: n.score)
            return self._generate_final_answer(problem, best), best.path_to_root()

        return "Unable to solve within the step budget.", root.path_to_root()

    def _dfs_solve(
        self,
        problem: str,
        node:    ThoughtNode,
    ) -> Tuple[str, List[str]]:
        """
        Depth-first search: go deep on the most promising branch first.
        """
        if node.depth >= self.max_depth:
            answer = self._generate_final_answer(problem, node)
            return answer, node.path_to_root()

        candidates = self._generate_thoughts(problem, node)

        # Score all candidates and sort
        scored = []
        for thought in candidates:
            child = ThoughtNode(thought=thought, depth=node.depth + 1, parent=node)
            child.score = self._evaluate_thought(problem, child)
            scored.append(child)
            node.children.append(child)

            if self._is_terminal(thought):
                return self._extract_answer(thought), child.path_to_root()

        # Recurse on the best candidate
        scored.sort(key=lambda n: n.score, reverse=True)
        if scored:
            return self._dfs_solve(problem, scored[0])

        return "Unable to find a solution.", node.path_to_root()

    def _generate_thoughts(self, problem: str, node: ThoughtNode) -> List[str]:
        """Ask the LLM to generate k candidate next thoughts."""
        path    = "\n".join(node.path_to_root())
        prompt  = (
            f"Problem: {problem}\n\n"
            f"Reasoning so far:\n{path}\n\n"
            f"Generate {self.n_thoughts} different possible next reasoning steps. "
            f"Number each one: 1. 2. 3.\n\n1."
        )

        raw = self.generator.generate(
            prompt=prompt,
            max_new_tokens=self.max_tokens * self.n_thoughts,
            strategy="top_p",
            temperature=0.7,
        )

        # Parse numbered thoughts
        thoughts = []
        for i in range(1, self.n_thoughts + 1):
            marker = f"{i}."
            next_m = f"{i + 1}."
            start  = raw.find(marker)
            end    = raw.find(next_m) if i < self.n_thoughts else len(raw)
            if start >= 0:
                thought = raw[start + len(marker):end].strip()
                if thought:
                    thoughts.append(f"Step {node.depth + 1}: {thought[:300]}")

        # Fallback: split by newlines if numbered parsing failed
        if not thoughts:
            thoughts = [
                f"Step {node.depth + 1}: {line.strip()}"
                for line in raw.split("\n")
                if line.strip() and len(line.strip()) > 10
            ][:self.n_thoughts]

        return thoughts or [f"Step {node.depth + 1}: Continue reasoning about the problem."]

    def _evaluate_thought(self, problem: str, node: ThoughtNode) -> float:
        """
        Ask the LLM to score a thought on a 1-10 scale.
        Returns a float in [0, 1].
        """
        path   = "\n".join(node.path_to_root())
        prompt = (
            f"Problem: {problem}\n\n"
            f"Reasoning path:\n{path}\n\n"
            f"On a scale of 1 to 10, how promising is this reasoning for solving the problem? "
            f"Reply with just a number.\nScore:"
        )

        raw = self.generator.generate(
            prompt=prompt,
            max_new_tokens=5,
            strategy="greedy",
        )

        try:
            import re
            nums = re.findall(r"\d+", raw)
            score = float(nums[0]) / 10.0 if nums else 0.5
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5

    def _is_terminal(self, thought: str) -> bool:
        """Check if a thought contains a final answer."""
        lower = thought.lower()
        return any(
            phrase in lower for phrase in [
                "final answer:", "therefore, the answer is",
                "the answer is", "in conclusion,", "we conclude"
            ]
        )

    def _extract_answer(self, thought: str) -> str:
        """Extract the final answer text from a terminal thought."""
        import re
        for pattern in [
            r"(?:final answer|the answer is|therefore)[:\s]+(.+?)(?:\.|$)",
        ]:
            match = re.search(pattern, thought, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return thought[:200]

    def _generate_final_answer(self, problem: str, node: ThoughtNode) -> str:
        """Ask the LLM to produce a final answer based on the reasoning path."""
        path = "\n".join(node.path_to_root())
        prompt = (
            f"Problem: {problem}\n\n"
            f"Reasoning:\n{path}\n\n"
            f"Based on this reasoning, provide the final answer:\nFinal Answer:"
        )
        return self.generator.generate(prompt, max_new_tokens=200, strategy="greedy")
