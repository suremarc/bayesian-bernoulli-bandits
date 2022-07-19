use rand::{
    distributions::Bernoulli,
    prelude::{Distribution, ThreadRng},
    Rng,
};

#[derive(Debug, Clone, Copy)]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn posterior(mut self, outcome: bool) -> Self {
        if outcome {
            self.alpha += 1.;
        } else {
            self.beta += 1.;
        }

        self
    }

    pub fn mean(self) -> f64 {
        self.alpha / (self.alpha + self.beta)
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
pub struct Belief<const N: usize> {
    pub state: Box<State<N>>,
    r: ThreadRng,
    pub dist: [Beta; N],
}

impl<const K: usize> Belief<K> {
    pub fn new(r: ThreadRng, prior: Beta) -> Self {
        let state = State::new_rand(r.clone());
        Belief {
            state: Box::new(state),
            r,
            dist: [prior; K],
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

    pub fn expected_reward(&self, action: usize) -> f64 {
        self.dist[action].mean()
    }

    fn bellman_recurse(&self, actions: &mut [usize], outcomes: &mut [bool]) -> f64 {
        if actions.is_empty() {
            return 0.;
        }

        let mut best: f64 = f64::MIN;
        for action in 0..K {
            let result = self.expected_reward(action)
                + self.bellman_recurse(&mut actions[1..], &mut outcomes[1..]);
            if result > best {
                actions[0] = action;
                best = result;
            }
        }

        best
    }

    pub fn best(&self, n: usize) -> usize {
        let s0 = (*self).clone();
        let mut actions: Vec<usize> = vec![0; n];
        let mut outcomes: Vec<bool> = vec![false; n];

        s0.bellman_recurse(actions.as_mut_slice(), outcomes.as_mut_slice());

        actions[0]
    }
}

fn main() {
    let mut r = ThreadRng::default();
    let mut belief = Belief::<4>::new(
        r.clone(),
        Beta {
            alpha: 1.,
            beta: 1.,
        },
    );

    println!("{:#?}", belief.state.p);

    let mut average_score: f64 = 0.;
    const I: usize = 1000;
    const N: usize = 10;
    for i in 0..I {
        let mut score = 0;
        for n in 0..N {
            let a = belief.best(N - n);
            let outcome = belief.take(a);
            score += outcome as u64;
            if i == 0 {
                println!("{a} -> {outcome} ({score})");
            }
        }

        average_score += score as f64;
    }
    average_score /= I as f64;

    println!("{average_score}");
    println!(
        "{:#?}",
        belief
            .state
            .p
            .as_ref()
            .iter()
            .map(|&p| 1. / ((I * N) as f64 * p * (1. - p)).sqrt())
            .collect::<Vec<f64>>()
    )

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
