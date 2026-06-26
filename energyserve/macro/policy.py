"""EAPS priority scoring for the macro-scheduler (Equation 8).

Isolates the Expected-cost-guided Anti-starvation Priority Selection score from
the queue mechanics. The score combines two terms:

    S(r, t) = E[W(r)]  -  beta * Psi(t - a_r)
              \\_______/    \\______________/
            expected cost      temporal decay (anti-starvation)

The first term ranks by the *expected* computational cost predicted by the
profiler (here, the predicted output length), exposing deceptive short-input /
long-output requests before admission. The second term lifts long-waiting
requests so none is starved. Lower score == higher priority.

Keeping the score pure (a function of a request and the current time) makes the
policy trivially unit-testable and decoupled from queue management.
"""


class EAPSPolicy:
    """Computes the EAPS priority score (Eq. 8); lower is higher priority."""

    def __init__(self, config):
        # beta: aging weight balancing throughput (cost) against fairness (age).
        self.aging_factor = config["scheduler"]["aging_factor"]

    def score(self, req, now):
        """Priority score for ``req`` at wall-clock time ``now``.

        Args:
            req: a request exposing ``expected_out_len`` and ``enqueue_time``.
            now (float): current wall-clock time (seconds).
        """
        expected_cost = req.expected_out_len
        temporal_decay = (now - req.enqueue_time) * self.aging_factor
        return expected_cost - temporal_decay
