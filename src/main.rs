#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]

use num_bigint::BigUint;
use num_traits::PrimInt;
use ordered_float::OrderedFloat;
use rand::{
    distributions::Bernoulli,
    prelude::{Distribution, ThreadRng},
    Rng,
};
use std::{borrow::Borrow, collections::BTreeMap, ops::AddAssign};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Observations<I: PrimInt + AddAssign> {
    pub successes: I,
    pub failures: I,
}

impl<I: PrimInt + AddAssign> Default for Observations<I> {
    fn default() -> Self {
        Self {
            successes: I::zero(),
            failures: I::zero(),
        }
    }
}

impl<I: PrimInt + AddAssign> Observations<I> {
    #[inline(always)]
    fn update(mut self, outcome: bool) -> Self {
        if outcome {
            self.successes += I::one();
        } else {
            self.failures += I::one();
        }

        self
    }

    #[inline(always)]
    fn n(self) -> I {
        self.successes + self.failures
    }
}

fn compositions_recurse<I: PrimInt + AddAssign, const K: usize>(
    entries: &mut Vec<[I; K]>,
    n: I,
    k: I,
) where
    I: Into<usize>,
{
    if n.is_zero() {
        return;
    }

    if k.is_one() {
        let mut entry = [I::zero(); K];
        entry[0] = n;
        entries.push(entry);
        return;
    }

    let current_len = entries.len();
    compositions_recurse::<I, K>(entries, n - I::one(), k);
    for x in entries[current_len..].iter_mut() {
        x[(k - I::one()).into()] += I::one();
    }

    let current_len = entries.len();
    compositions_recurse::<I, K>(entries, n - I::one(), k - I::one());
    for x in entries[current_len..].iter_mut() {
        x[(k - I::one()).into()] = I::one();
    }
}

pub fn compositions_count<I: PrimInt + AddAssign, const K: usize>(n: I) -> usize
where
    I: Into<BigUint>,
    I: Into<usize>,
{
    num_integer::binomial::<BigUint>(
        (n - I::one()).into(),
        (I::from(K).unwrap() - I::one()).into(),
    )
    .try_into()
    .unwrap()
}

pub fn compositions<I: PrimInt + AddAssign, const K: usize>(n: I) -> Vec<[I; K]>
where
    I: Into<BigUint>,
    I: Into<usize>,
{
    let mut entries = Vec::with_capacity(compositions_count::<I, K>(n));
    compositions_recurse::<I, K>(&mut entries, n, I::from(K).unwrap());

    entries
}

pub fn weak_compositions<I: PrimInt + AddAssign, const K: usize>(n: I) -> Vec<[I; K]>
where
    I: Into<BigUint>,
    I: Into<usize>,
{
    let mut result = compositions::<I, K>(n + I::from(K).unwrap());
    for entry in result.iter_mut() {
        *entry = entry.map(|x| x - I::one());
    }

    result
}

pub fn enumerate_observations<I: PrimInt + AddAssign, const K: usize>(
    n: I,
) -> Vec<[Observations<I>; K]>
where
    [(); K * 2]:,
    I: Into<BigUint>,
    I: Into<usize>,
{
    weak_compositions::<I, { K * 2 }>(n)
        .into_iter()
        .map(|cell| {
            let mut betas: [Observations<I>; K] = [Default::default(); K];
            for (i, &[successes, failures]) in unsafe { cell.as_chunks_unchecked::<2>() }
                .iter()
                .enumerate()
            {
                betas[i] = Observations {
                    successes,
                    failures,
                }
            }

            betas
        })
        .collect()
}

pub fn enumerate_observations_recursive<I: PrimInt + AddAssign, const K: usize>(
    n: I,
) -> Vec<[Observations<I>; K]>
where
    [(); K * 2]:,
    I: Into<BigUint>,
    I: Into<usize>,
{
    (0..=n.into())
        .rev()
        .map(|n| I::from(n).unwrap())
        .flat_map(enumerate_observations::<I, K>)
        .collect()
}

#[derive(Debug, Clone)]
pub struct State<const K: usize> {
    pub p: [f64; K],
    dist: [Bernoulli; K],
}

impl<const K: usize> State<K> {
    pub fn new(p: [f64; K]) -> Self {
        let dist = p.map(|p| Bernoulli::new(p).unwrap());
        Self { p, dist }
    }
    pub fn new_rand(r: &mut ThreadRng) -> Self {
        let p: [f64; K] = (0..K)
            .map(|_| r.gen_range(0.0..1.0))
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap();

        Self::new(p)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Prior {
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Belief<I: PrimInt + AddAssign, const K: usize>(pub [Observations<I>; K]);

impl<I: PrimInt + AddAssign, const K: usize> Belief<I, K>
where
    I: Into<f64>,
{
    pub fn take(&mut self, action: usize, state: &State<K>, r: &mut ThreadRng) -> bool {
        let outcome = state.dist[action].sample(r);
        self.0[action] = self.0[action].update(outcome);
        outcome
    }

    fn n(&self) -> I {
        let mut sum = I::zero();
        for obs in self.0 {
            sum += obs.n();
        }

        sum
    }

    #[inline(always)]
    fn sample_beta(&self, action: usize, prior: Prior, r: &mut ThreadRng) -> f64 {
        let alpha_recip = (prior.alpha + self.0[action].successes.into()).recip();
        let beta_recip = (prior.beta + self.0[action].failures.into()).recip();
        let (mut rv1, mut rv2): (f64, f64) = Default::default();

        loop {
            rv1 = r.gen::<f64>().powf(alpha_recip);
            rv2 = r.gen::<f64>().powf(beta_recip);

            if rv1 + rv2 <= 1. {
                return rv1 / (rv1 + rv2);
            }
        }
    }

    #[inline(always)]
    fn success_chance(&self, action: usize, prior: Prior) -> f64 {
        let dist = self.0[action];
        (prior.alpha + dist.successes.into())
            / ((dist.successes + dist.failures).into() + prior.alpha + prior.beta)
    }

    #[inline]
    fn possible_transitions(&self, action: usize, prior: Prior) -> [(f64, Self); 2] {
        let p = self.success_chance(action, prior);
        [(true, p), (false, 1. - p)].map(|(outcome, prob)| {
            let mut new = self.clone();
            new.0[action] = new.0[action].update(outcome);
            (prob, new)
        })
    }

    pub fn value_iteration(
        n: I,
        epsilon: f64,
        prior: Prior,
        orig: impl Fn(&Self) -> f64,
    ) -> BTreeMap<Self, f64>
    where
        [(); K * 2]:,
        I: Into<BigUint>,
        I: Into<usize>,
    {
        let states = enumerate_observations_recursive::<I, K>(n)
            .into_iter()
            .map(|state| Self(state))
            .collect::<Vec<_>>();
        let mut value: BTreeMap<&Self, f64> =
            states.iter().map(|state| (state, orig(state))).collect();

        loop {
            let mut delta: f64 = 0.;
            for state in states.iter() {
                let mut new_reward = 0.;
                if n == state.n() {
                    new_reward += state.best_action_from_map(&value, prior).1;
                }

                let old_reward = value.insert(state, new_reward).unwrap();
                delta = delta.max((new_reward - old_reward).abs());
            }

            if delta < epsilon {
                break;
            }
        }

        value
            .into_iter()
            .map(|(belief, reward)| (belief.clone(), reward))
            .collect()
    }

    fn dynamic_search_recurse(&self, n: I, prior: Prior, value: &mut BTreeMap<Self, f64>) -> f64 {
        if n.is_zero() {
            return 0.;
        }

        match value.get(self).copied() {
            Some(reward) => reward,
            None => {
                let (_, reward) = self.best_action_with_lookahead(
                    |new_belief| new_belief.dynamic_search_recurse(n - I::one(), prior, value),
                    prior,
                );

                value.insert(self.clone(), reward);
                reward
            }
        }
    }

    pub fn dynamic_search(&self, n: I, prior: Prior) -> BTreeMap<Self, f64>
    where
        I: Into<usize>,
    {
        let mut value = BTreeMap::new();
        self.dynamic_search_recurse(n, prior, &mut value);

        value
    }

    pub fn best_action_epsilon_greedy(
        &self,
        epsilon: f64,
        prior: Prior,
        r: &mut ThreadRng,
    ) -> (usize, f64) {
        if r.gen::<f64>() > epsilon {
            self.best_action_with_lookahead(|_| 0., prior)
        } else {
            let action = r.gen_range(0..K);
            (action, self.success_chance(action, prior))
        }
    }

    pub fn best_action_ucb(&self, n: I, prior: Prior) -> (usize, f64) {
        (0..K)
            .map(|action| {
                (
                    action,
                    self.success_chance(action, prior)
                        + (2. * (n.into()).ln() / (self.0[action].n().into())).sqrt(),
                )
            })
            .max_by_key(|&(_, reward)| OrderedFloat::from(reward))
            .unwrap()
    }

    pub fn best_action_thompson(&self, prior: Prior, r: &mut ThreadRng) -> (usize, f64) {
        (0..K)
            .map(|action| (action, self.sample_beta(action, prior, r)))
            .max_by_key(|&(_, reward)| OrderedFloat::from(reward))
            .unwrap()
    }

    fn best_action_from_map<T>(&self, value: &BTreeMap<T, f64>, prior: Prior) -> (usize, f64)
    where
        T: Borrow<Self> + Ord,
    {
        self.best_action_with_lookahead(
            |new_belief| value.get(new_belief.borrow()).copied().unwrap_or(0.),
            prior,
        )
    }

    fn best_action_with_lookahead(
        &self,
        mut lookahead: impl FnMut(&Self) -> f64,
        prior: Prior,
    ) -> (usize, f64) {
        (0..K)
            .map(|action| {
                (
                    action,
                    self.success_chance(action, prior)
                        + self
                            .possible_transitions(action, prior)
                            .map(|(prob, new_belief)| prob * lookahead(&new_belief))
                            .iter()
                            .sum::<f64>(),
                )
            })
            .max_by_key(|&(_, reward)| OrderedFloat::from(reward))
            .unwrap()
    }
}

fn main() {
    const K: usize = 3;
    type Size = u16;
    const N: Size = 100;

    let mut r = ThreadRng::default();
    let state = State::<K>::new_rand(&mut r);
    // let state = State::<K>::new([0.8341146271245636, 0.7968395308492424]);
    eprintln!("{:#?}", state.p);
    let prior = Prior {
        alpha: 1.,
        beta: 1.,
    };

    let mut belief = Belief::<Size, K>([Default::default(); K]);

    let t0 = std::time::Instant::now();
    // let value = Belief::value_iteration(N, 5., prior, |_| 0.);
    // let value = belief.dynamic_search(N, prior);

    let mut score = 0;
    for n in 0..N {
        // let (action, expected_reward) = belief.best_action_from_map(&value, prior);
        // let (action, expected_reward) = belief.best_action_epsilon_greedy(0.05, prior, &mut r);
        // let (action, expected_reward) = belief.best_action_ucb(n, prior);
        let (action, expected_reward) = belief.best_action_thompson(prior, &mut r);

        let expected_score = score as f64 + expected_reward;
        let outcome = belief.take(action, &state, &mut r);
        score += outcome as u64;
        // eprintln!("{n}: ({action}, {expected_score:.2}) -> {outcome} ({score})");
    }

    eprintln!(
        "# of states: idk ({}ms)\n got {:.2}%",
        // value.len(),
        // value.len(),
        t0.elapsed().as_millis(),
        score as f64 / (N as f64) * 100.,
    );
}
