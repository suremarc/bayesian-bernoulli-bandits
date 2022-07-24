#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::PrimInt;
use ordered_float::OrderedFloat;
use rand::{distributions::Bernoulli, prelude::ThreadRng, Rng};
use std::{collections::BTreeMap, ops::AddAssign};

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
    let mut entries = Vec::with_capacity(compositions_count::<I, K>(n.try_into().unwrap()));
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
    p: [f64; K],
    dist: [Bernoulli; K],
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

pub struct Params<const K: usize> {
    state: State<K>,
    alpha: f64,
    beta: f64,
}

pub struct Belief<'a, I: PrimInt + AddAssign, const K: usize> {
    params: &'a Params<K>,
    dist: [Observations<I>; K],
}

impl<'a, I: PrimInt + AddAssign, const K: usize> Belief<'a, I, K>
where
    I: Into<f64>,
{
    pub fn new(params: &'a Params<K>) -> Self {
        Self {
            params,
            dist: [Default::default(); K],
        }
    }

    #[inline(always)]
    fn immediate_reward(&self, action: usize) -> f64 {
        let dist = self.dist[action];
        (self.params.alpha + dist.successes.into())
            / ((dist.successes + dist.failures).into() + self.params.alpha + self.params.beta)
    }
}

fn main() {}
