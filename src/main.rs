#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]

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

pub fn betas<const K: usize>(n: usize) -> Vec<[Beta; K]>
where
    [(); K * 2]:,
{
    weak_compositions::<{ K * 2 }>(n.try_into().unwrap())
        .into_iter()
        .map(|cell| {
            let mut betas: [Beta; K] = [Default::default(); K];
            for (i, &[successes, failures]) in unsafe { cell.as_chunks_unchecked::<2>() }
                .iter()
                .enumerate()
            {
                betas[i] = Beta {
                    successes,
                    failures,
                }
            }

            betas
        })
        .collect()
}

pub fn betas_recursive<const K: usize>(n: usize) -> Vec<[Beta; K]>
where
    [(); K * 2]:,
{
    (0..=n).rev().flat_map(betas::<K>).collect()
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

    #[inline(always)]
    fn expected_reward<F>(&self, action: usize, mut lookahead: F) -> f64
    where
        F: FnMut(&[Beta; K]) -> f64,
    {
        let p_success = self.dist[action].mean(self.alpha, self.beta);
        let mut reward = p_success;
        for (outcome, prob) in [(true, p_success), (false, 1. - p_success)] {
            let mut dist = self.dist;
            dist[action] = dist[action].posterior(outcome);
            reward += prob * lookahead(&dist);
        }

        reward
    }

    pub fn value_iterate(&self, n: usize, tolerance: f64) -> BTreeMap<[Beta; K], f64>
    where
        [(); K * 2]:,
    {
        let states = betas_recursive::<K>(n);
        println!("{states:#?}");
        let mut value: BTreeMap<&[Beta; K], f64> = states.iter().map(|beta| (beta, 0.)).collect();
        // println!("{value:#?}");

        let mut delta: f64 = 0.;
        loop {
            for state in states.iter() {
                let new = (0..K)
                    .map(|action| self.expected_reward(action, |dist| *value.get(dist).unwrap()))
                    .max_by_key(|&reward| OrderedFloat::from(reward))
                    .unwrap();
                let orig = value.insert(state, new).unwrap();

                delta = delta.max((new - orig).abs());
            }

            if delta < tolerance {
                return value
                    .into_iter()
                    .map(|(&beta, value)| (beta, value))
                    .collect();
            }
            println!("{delta}");
            println!("{value:#?}");
        }
    }

    pub fn best_action(&self, value: &BTreeMap<[Beta; K], f64>) -> usize {
        (0..K)
            .map(|action| {
                (
                    action,
                    self.expected_reward(action, |dist| *value.get(dist).unwrap()),
                )
            })
            .max_by_key(|&(_, reward)| OrderedFloat::from(reward))
            .unwrap()
            .0
    }

    fn bellman_recurse(
        &self,
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
                        (
                            action,
                            self.expected_reward(action, |dist| {
                                Self::bellman_recurse(self, dist, n - 1, memoized)
                            }),
                        )
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
        self.bellman_recurse(&self.dist, n, memoized);
        memoized.get(&self.dist).copied()
    }
}

fn main() {
    let r = ThreadRng::default();
    let state = State::new_rand(r.clone());
    eprintln!("{:#?}", state.p);

    let mut average_score: f64 = 0.;
    const I: usize = 1;
    const N: usize = 10;
    for i in 0..I {
        let mut belief = Belief::<2>::new(&state, r.clone(), 1., 1.);

        let mut memoized = BTreeMap::new();
        // let value = belief.value_iterate(N, 1.);
        let t0 = std::time::Instant::now();

        let mut score = 0;
        for n in 0..N {
            let (a, expected_reward) = belief.best_with_memoized(N - n, &mut memoized).unwrap();
            // let a = belief.best_action(&value);
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
                // value.len(),
                t0.elapsed().as_millis()
            );
        }

        average_score += score as f64;
    }
    average_score /= I as f64;

    eprintln!("{average_score}");
}
