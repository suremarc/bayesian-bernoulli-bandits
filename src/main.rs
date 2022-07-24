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
    n: usize,
) -> Vec<[Observations<I>; K]>
where
    [(); K * 2]:,
    I: Into<BigUint>,
    I: Into<usize>,
{
    (0..=n)
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
    pub fn new_rand(r: &mut ThreadRng) -> State<K> {
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

#[derive(Debug, Clone, Copy)]
pub struct Params {
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

    #[inline(always)]
    fn success_chance(&self, action: usize, params: Params) -> f64 {
        let dist = self.0[action];
        (params.alpha + dist.successes.into())
            / ((dist.successes + dist.failures).into() + params.alpha + params.beta)
    }

    #[inline]
    fn possible_transitions(&self, action: usize, params: Params) -> [(f64, Self); 2] {
        let p = self.success_chance(action, params);
        [(true, p), (false, 1. - p)].map(|(outcome, prob)| {
            let mut new = self.clone();
            new.0[action] = new.0[action].update(outcome);
            (prob, new)
        })
    }

    pub fn value_iteration(&self, n: I, epsilon: f64, params: Params) -> BTreeMap<Self, f64>
    where
        [(); K * 2]:,
        I: Into<BigUint>,
        I: Into<usize>,
        I: std::fmt::Debug,
    {
        let states = enumerate_observations::<I, K>(n)
            .into_iter()
            .map(|state| Self(state))
            .collect::<Vec<_>>();
        let mut value: BTreeMap<&Self, f64> = states.iter().map(|state| (state, 0.)).collect();

        let mut delta: f64 = 1.;
        loop {
            for state in states.iter() {
                let new_reward = self.best_action_and_reward(&value, params).1;

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

    fn best_action_and_reward<T>(&self, value: &BTreeMap<T, f64>, params: Params) -> (usize, f64)
    where
        T: Borrow<Self> + Ord,
        I: std::fmt::Debug,
    {
        (0..K)
            .map(|action| {
                (
                    action,
                    self.success_chance(action, params)
                        + self
                            .possible_transitions(action, params)
                            .map(|(prob, new_state)| {
                                println!("{new_state:#?}");
                                prob * *value.get(new_state.borrow()).unwrap()
                            })
                            .iter()
                            .sum::<f64>(),
                )
            })
            .max_by_key(|&(_, reward)| OrderedFloat::from(reward))
            .unwrap()
    }
}

fn main() {
    const K: usize = 2;
    const N: u8 = 10;

    let mut r = ThreadRng::default();
    let state = State::<K>::new_rand(&mut r);
    let params = Params {
        alpha: 1.,
        beta: 1.,
    };

    let mut belief = Belief::<u8, K>([Default::default(); K]);

    let t0 = std::time::Instant::now();
    let value = belief.value_iteration(N, 1., params);

    let mut score = 0;
    for n in 0..N {
        let (action, expected_reward) = belief.best_action_and_reward(&value, params);

        let expected_score = score as f64 + expected_reward;
        let outcome = belief.take(action, &state, &mut r);
        score += outcome as u64;
        eprintln!("{n}: ({action}, {expected_score:.2}) -> {outcome} ({score})");
    }

    eprintln!(
        "# of states: {} ({}ms)",
        value.len(),
        // value.len(),
        t0.elapsed().as_millis()
    );
}
