use num_bigint::BigUint;
use ordered_float::OrderedFloat;
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

fn compositions_recurse<const K: usize>(entries: &mut Vec<[u8; K]>, n: u8, k: u8) {
    if n == 0 {
        return;
    }

    if k == 1 {
        let mut entry = [0; K];
        entry[0] = n;
        entries.push(entry);
        return;
    }

    let current_len = entries.len();
    compositions_recurse::<K>(entries, n - 1, k);
    for x in entries[current_len..].iter_mut() {
        x[(k - 1) as usize] += 1;
    }

    let current_len = entries.len();
    compositions_recurse::<K>(entries, n - 1, k - 1);
    for x in entries[current_len..].iter_mut() {
        x[(k - 1) as usize] = 1;
    }
}

pub fn compositions_count<const K: usize>(n: u8) -> usize {
    num_integer::binomial(BigUint::from(n as usize - 1), BigUint::from(K - 1))
        .try_into()
        .unwrap()
}

pub fn compositions<const K: usize>(n: u8) -> Vec<[u8; K]> {
    let mut entries = Vec::with_capacity(compositions_count::<K>(n));
    compositions_recurse::<K>(&mut entries, n, K.try_into().unwrap());

    entries
}

pub fn weak_compositions<const K: usize>(n: u8) -> Vec<[u8; K]> {
    let mut result = compositions::<K>(n + u8::try_from(K).unwrap());
    for entry in result.iter_mut() {
        *entry = entry.map(|x| x - 1);
    }

    result
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
                let (best_action, reward) = (0..K)
                    .map(|action| {
                        let p_success = dist[action].mean(alpha, beta);
                        let mut reward = p_success;
                        for (outcome, prob) in [(true, p_success), (false, 1. - p_success)] {
                            let mut dist = *dist;
                            dist[action] = dist[action].posterior(outcome);
                            reward +=
                                prob * Self::bellman_recurse(alpha, beta, &dist, n - 1, memoized);
                        }

                        (action, reward)
                    })
                    .max_by_key(|&(_, reward)| OrderedFloat::from(reward))
                    .unwrap();

                memoized.insert(*dist, (best_action, reward));
                reward
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
}
