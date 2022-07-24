use rand::{distributions::Bernoulli, prelude::ThreadRng, Rng};
use std::collections::BTreeMap;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Beta {
    pub successes: u8,
    pub failures: u8,
}

impl Beta {
    #[inline(always)]
    fn posterior(mut self, outcome: bool) -> Self {
        if outcome {
            self.successes += 1;
        } else {
            self.failures += 1
        }

        self
    }

    #[inline(always)]
    fn mean(self, alpha: f64, beta: f64) -> f64 {
        (alpha + self.successes as f64) / ((self.successes + self.failures) as f64 + alpha + beta)
    }
}

#[derive(Debug, Clone)]
pub struct State<const K: usize> {
    pub p: [f64; K],
    pub dist: [Bernoulli; K],
}

impl<const K: usize> State<K> {
    pub fn new_rand(mut r: ThreadRng) -> State<K> {
        let p: [f64; K] = (0..K)
            .map(|_| r.gen_range(0.0..1.0))
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap();
        let dist = p
            .iter()
            .map(|&p| Bernoulli::new(p).unwrap())
            .collect::<Vec<Bernoulli>>()
            .try_into()
            .unwrap();
        State { p, dist }
    }
}

#[derive(Debug, Clone)]
pub struct Belief<'a, const K: usize> {
    pub state: &'a State<K>,
    r: ThreadRng,
    pub dist: [Beta; K],
    pub alpha: f64,
    pub beta: f64,
}

impl<'a, const K: usize> Belief<'a, K> {
    pub fn new(state: &'a State<K>, r: ThreadRng, alpha: f64, beta: f64) -> Self {
        Belief {
            state,
            r,
            dist: [Default::default(); K],
            alpha,
            beta,
        }
    }

    pub fn posterior(&mut self, action: usize, outcome: bool) {
        self.dist[action] = self.dist[action].posterior(outcome);
    }

    pub fn take(&mut self, action: usize) -> bool {
        let outcome = self.r.gen_bool(self.state.p[action]);
        // println!("{outcome}");
        self.posterior(action, outcome);

        outcome
    }

    fn bellman_recurse(
        alpha: f64,
        beta: f64,
        dist: &[Beta; K],
        n: usize,
        memoized: &mut BTreeMap<[Beta; K], (usize, f64)>,
    ) -> f64 {
        if n == 0 {
            return 0.;
        }

        match memoized.get(dist) {
            Some(&(_, reward)) => reward,
            None => {
                // println!("1");
                let mut best = f64::MIN;
                let mut best_action = 0_usize;
                for action in 0..K {
                    let p_success = dist[action].mean(alpha, beta);
                    let mut reward = p_success;
                    for (outcome, prob) in [(true, p_success), (false, 1. - p_success)] {
                        let mut dist = *dist;
                        dist[action] = dist[action].posterior(outcome);
                        reward += prob * Self::bellman_recurse(alpha, beta, &dist, n - 1, memoized);
                    }

                    if reward > best {
                        best = reward;
                        best_action = action;
                    }
                }

                memoized.insert(*dist, (best_action, best));
                best
            }
        }
    }

    pub fn best(&self, n: usize) -> Option<(usize, f64)> {
        let mut memoized: BTreeMap<[Beta; K], (usize, f64)> = BTreeMap::new();
        self.best_with_memoized(n, &mut memoized)
    }

    pub fn best_with_memoized(
        &self,
        n: usize,
        memoized: &mut BTreeMap<[Beta; K], (usize, f64)>,
    ) -> Option<(usize, f64)> {
        Self::bellman_recurse(self.alpha, self.beta, &self.dist, n, memoized);
        memoized.get(&self.dist).copied()
    }
}

fn main() {
    let r = ThreadRng::default();
    let state = State::new_rand(r.clone());
    eprintln!("{:#?}", state.p);

    let mut average_score: f64 = 0.;
    const I: usize = 1;
    const N: usize = 100;
    for i in 0..I {
        let mut belief = Belief::<2>::new(&state, r.clone(), 1., 1.);

        let mut memoized = BTreeMap::new();
        let t0 = std::time::Instant::now();

        let mut score = 0;
        for n in 0..N {
            let (a, expected_reward) = belief.best_with_memoized(N - n, &mut memoized).unwrap();
            let expected_score = score as f64 + expected_reward;
            let outcome = belief.take(a);
            score += outcome as u64;
            if i == 0 {
                eprintln!("{n}: ({a}, {expected_score:.2}) -> {outcome} ({score})");
            }
        }

        if i == 0 {
            eprintln!(
                "# of states: {} ({}ms)",
                memoized.len(),
                t0.elapsed().as_millis()
            );
        }

        average_score += score as f64;
    }
    average_score /= I as f64;

    eprintln!("{average_score}");
    // println!(
    //     "{:#?}",
    //     state
    //         .p
    //         .as_ref()
    //         .iter()
    //         .map(|&p| 1. / ((I * N) as f64 * p * (1. - p)).sqrt())
    //         .collect::<Vec<f64>>()
    // )

    // let mut sum = 0.;
    // const N: usize = 1000;
    // for _ in 0..N {
    //     let realized_reward: u64 = actions
    //         .iter()
    //         .map(|&a| belief.state.dist[a].sample(&mut r) as u64)
    //         .sum();
    //     // println!("{}", realized_reward);
    //     sum += realized_reward as f64;
    // }
    // sum /= N as f64;
    // println!("average: {}", sum);
}
