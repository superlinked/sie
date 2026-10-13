//! [`Lane`]: when a model's pending work is handed to its engine.
//!
//! A [`BatchFormer`](super::BatchFormer) decides *what* goes into a batch.
//! A lane decides *when* the next batch leaves, for an engine that runs
//! one batch at a time. The engine is the scarce thing: a batch sent
//! while another is still with it only waits at the engine's lock, with
//! its membership frozen, and the requests that arrive a moment later
//! start a batch of their own behind it. Two groups then take turns and
//! every request waits for the other group's forward.
//!
//! The rule:
//!
//! 1. **Idle lane, send at once.** Nothing is with the engine and no batch
//!    has just come back: a request leaves immediately with everything
//!    else pending. A lone request never waits a window.
//! 2. **Busy lane, hold.** While a batch is with the engine, arrivals
//!    stay in the queue where later arrivals can join them. Two things go
//!    ahead into a free pipeline slot instead: the next batch if it is
//!    already full, since holding it cannot improve it, and the rest of a
//!    request whose first part is with the engine, since its caller is
//!    waiting on that part and has nobody to come back with. "Next" is the
//!    scheduler's first-come-first-served pick across its queues: a full
//!    batch on a younger queue does not overtake an older queue's held
//!    one, which would then wait a forward longer for it.
//! 3. **On a return, gather once.** The callers just answered are the
//!    ones most likely to come straight back. The lane waits for as many
//!    arrivals as the returned batch answered, then sends everything
//!    pending as one batch. It stops waiting when nothing has arrived for
//!    the coalesce window, and in any case after half the returned
//!    batch's round trip or the batch wait, whichever is shorter. A batch
//!    that answered nobody (the first part of a request a cap cut) has no
//!    caller to wait for.
//! 4. **Skip the gather where it does not pay.** The lane scores whether
//!    returns are followed by that many arrivals inside the window. A
//!    closed loop of callers scores high; arrivals at random times score
//!    low, and then a return is followed by an immediate send. The score
//!    keeps updating either way, so the lane notices when callers start
//!    coming back.
//!
//! Everything here is synchronous and takes the current time as an
//! argument, so the rule is tested without a clock or a runtime. The
//! async plumbing is [`Scheduler::next_dispatch`](super::Scheduler).

use std::time::{Duration, Instant};

use super::batch_former::FlushReason;

/// Share of a returned batch's round trip the lane may spend gathering.
/// Half a forward is the most a caller should wait for company: past
/// that, two forwards one after another beat one late forward.
const GATHER_ROUND_TRIP_SHARE: f64 = 0.5;

/// A gather window shorter than this is not worth a timer.
const MIN_GATHER: Duration = Duration::from_micros(200);

/// Weight of the newest observation in the echo score.
const ECHO_SCORE_WEIGHT: f64 = 0.25;

/// The lane gathers while the echo score is at or above this.
const ECHO_SCORE_GATHER_AT: f64 = 0.5;

/// What the lane says about the pending work right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Send the next batch now, for this reason.
    Send(FlushReason),
    /// Nothing to do before this instant unless something arrives or a
    /// batch returns.
    WaitUntil(Instant),
    /// Wait for a batch to return (or for the pending work to fill up).
    Hold,
}

/// Arrivals expected after a returned batch, and the time they have.
#[derive(Debug, Clone, Copy)]
struct EchoWatch {
    /// No waiting past this instant, whatever arrives.
    latest: Instant,
    /// Waiting also ends when nothing has arrived for this long.
    quiet: Duration,
    /// The return, or the last arrival since.
    last_event: Instant,
    needed: usize,
    seen: usize,
}

impl EchoWatch {
    /// When the gather ends unless another request arrives first.
    fn deadline(&self) -> Instant {
        // A coalesce window too long to add to an instant never ends the
        // gather by itself; `latest` still does.
        self.last_event
            .checked_add(self.quiet)
            .map_or(self.latest, |quiet_end| self.latest.min(quiet_end))
    }
}

/// Dispatch state of one model's lane. One per
/// [`Scheduler`](super::Scheduler), shared by all its `(op, lora)` queues
/// because they share the engine.
#[derive(Debug)]
pub(crate) struct Lane {
    /// Batches handed to the engine and not yet returned.
    in_flight: usize,
    /// Open after a return until its arrivals came back or its window
    /// closed.
    watch: Option<EchoWatch>,
    /// Running share of returns whose callers came straight back.
    echo_score: f64,
    /// The pending work has waited in a gather since the last send.
    gathered: bool,
    /// The last batch sent left part of one of its requests in the queue.
    continues: bool,
}

impl Default for Lane {
    fn default() -> Self {
        Self {
            in_flight: 0,
            watch: None,
            // Start by gathering: a closed loop gains from its first
            // return, and callers that do not come back turn it off within
            // three returns.
            echo_score: 1.0,
            gathered: false,
            continues: false,
        }
    }
}

impl Lane {
    /// `requests` requests arrived.
    pub(crate) fn on_arrival(&mut self, now: Instant, requests: usize) {
        let Some(watch) = self.watch.as_mut() else {
            return;
        };
        if now >= watch.deadline() {
            self.settle_watch(now);
            return;
        }
        watch.seen += requests;
        watch.last_event = now;
        if watch.seen >= watch.needed {
            self.watch = None;
            self.record_echo(true);
        }
    }

    /// A batch was handed to the engine. `leaves_remainder` says a cap cut
    /// one of its requests and the rest of that request is still pending.
    pub(crate) fn on_dispatch(&mut self, leaves_remainder: bool) {
        self.in_flight += 1;
        self.gathered = false;
        self.continues = leaves_remainder;
    }

    /// A batch that answered `answered` requests came back after
    /// `round_trip`. `quiet` is the coalesce window and `longest` the batch
    /// wait: the gather ends when nothing has arrived for `quiet`, and no
    /// later than `longest` or half the round trip after the return.
    pub(crate) fn on_return(
        &mut self,
        now: Instant,
        answered: usize,
        round_trip: Duration,
        quiet: Duration,
        longest: Duration,
    ) {
        self.in_flight = self.in_flight.saturating_sub(1);
        if self.in_flight > 0 {
            // Still busy: the batch that empties the lane opens the watch.
            return;
        }
        // A watch left open by an earlier return is over either way.
        self.close_watch();
        let total = longest.min(round_trip.mul_f64(GATHER_ROUND_TRIP_SHARE));
        if answered == 0 || quiet.min(total) < MIN_GATHER {
            return;
        }
        self.watch = Some(EchoWatch {
            latest: now + total,
            quiet,
            last_event: now,
            needed: answered,
            seen: 0,
        });
    }

    /// What to do with the pending work. `full` is the cap it has already
    /// reached, if any. The caller holds a free pipeline slot.
    pub(crate) fn verdict(&mut self, now: Instant, full: Option<FlushReason>) -> Verdict {
        if let Some(reason) = full {
            return Verdict::Send(reason);
        }
        if self.in_flight > 0 {
            // The rest of a request that is partly with the engine follows
            // it at once; anything else waits for the return.
            return if self.continues {
                Verdict::Send(FlushReason::Drain)
            } else {
                Verdict::Hold
            };
        }
        self.settle_watch(now);
        match self.watch {
            Some(watch) if self.gathers() => {
                self.gathered = true;
                Verdict::WaitUntil(watch.deadline())
            }
            // A batch just returned and its callers have not been coming
            // back: send what is pending behind it.
            Some(_) => Verdict::Send(FlushReason::Drain),
            None if self.gathered => Verdict::Send(FlushReason::Coalesce),
            None => Verdict::Send(FlushReason::IdleBypass),
        }
    }

    /// Nothing is with the engine and no return is being watched.
    pub(crate) fn is_idle(&mut self, now: Instant) -> bool {
        self.settle_watch(now);
        self.in_flight == 0 && self.watch.is_none()
    }

    /// Batches with the engine.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// Whether a return is currently followed by a gather.
    pub(crate) fn gathers(&self) -> bool {
        self.echo_score >= ECHO_SCORE_GATHER_AT
    }

    /// Close a watch whose window has passed, as a miss.
    fn settle_watch(&mut self, now: Instant) {
        if self.watch.is_some_and(|watch| now >= watch.deadline()) {
            self.close_watch();
        }
    }

    /// Close the open watch, if any, scoring what it saw.
    fn close_watch(&mut self) {
        if let Some(watch) = self.watch.take() {
            self.record_echo(watch.seen >= watch.needed);
        }
    }

    fn record_echo(&mut self, came_back: bool) {
        let observation = if came_back { 1.0 } else { 0.0 };
        self.echo_score += ECHO_SCORE_WEIGHT * (observation - self.echo_score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COALESCE: Duration = Duration::from_millis(5);
    const LONGEST: Duration = Duration::from_millis(15);

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    fn us(value: u64) -> Duration {
        Duration::from_micros(value)
    }

    /// One closed-loop cycle with work pending: the batch returns at `t`,
    /// the loop asks for a verdict, then `callers` requests come back
    /// within a millisecond. Returns the verdict after the last arrival.
    fn return_then_echo(lane: &mut Lane, t: Instant, answered: usize, callers: usize) -> Verdict {
        lane.on_return(t, answered, ms(6), COALESCE, LONGEST);
        let _ = lane.verdict(t, None);
        for caller in 0..callers {
            lane.on_arrival(t + us(100 * (caller as u64 + 1)), 1);
        }
        lane.verdict(t + us(100 * callers as u64 + 1), None)
    }

    #[test]
    fn a_request_on_an_idle_lane_is_sent_at_once() {
        let mut lane = Lane::default();
        let now = Instant::now();
        lane.on_arrival(now, 1);
        assert_eq!(
            lane.verdict(now, None),
            Verdict::Send(FlushReason::IdleBypass)
        );
    }

    #[test]
    fn a_request_arriving_while_a_batch_is_out_is_held() {
        let mut lane = Lane::default();
        let now = Instant::now();
        lane.on_dispatch(false);
        lane.on_arrival(now + ms(1), 1);
        assert_eq!(lane.verdict(now + ms(1), None), Verdict::Hold);
        // However long it has waited: the engine is the limit, not a timer.
        assert_eq!(lane.verdict(now + ms(500), None), Verdict::Hold);
    }

    #[test]
    fn a_full_batch_goes_ahead_while_a_batch_is_out() {
        let mut lane = Lane::default();
        lane.on_dispatch(false);
        assert_eq!(
            lane.verdict(Instant::now(), Some(FlushReason::CountCap)),
            Verdict::Send(FlushReason::CountCap)
        );
    }

    #[test]
    fn after_a_return_the_lane_waits_for_the_callers_it_answered() {
        let mut lane = Lane::default();
        let t = Instant::now();
        lane.on_dispatch(false);
        // Seven requests arrived during the forward and are pending.
        lane.on_arrival(t, 7);
        lane.on_return(t + ms(6), 1, ms(6), COALESCE, LONGEST);
        // The one caller just answered has not come back yet: gather, for
        // at most half the round trip.
        assert_eq!(
            lane.verdict(t + ms(6), None),
            Verdict::WaitUntil(t + ms(6) + ms(3))
        );
        // It comes back: all eight leave together, with no timer involved.
        lane.on_arrival(t + ms(6) + us(400), 1);
        assert_eq!(
            lane.verdict(t + ms(6) + us(400), None),
            Verdict::Send(FlushReason::Coalesce)
        );
    }

    #[test]
    fn a_lone_caller_in_a_closed_loop_never_waits_the_window() {
        let mut lane = Lane::default();
        let mut t = Instant::now();
        for _ in 0..20 {
            lane.on_dispatch(false);
            t += ms(6);
            lane.on_return(t, 1, ms(6), COALESCE, LONGEST);
            // Nothing is pending until the caller is back, and its arrival
            // is the one the lane was waiting for.
            t += us(300);
            lane.on_arrival(t, 1);
            assert_eq!(
                lane.verdict(t, None),
                Verdict::Send(FlushReason::IdleBypass)
            );
        }
    }

    #[test]
    fn the_gather_ends_when_nothing_has_arrived_for_the_coalesce_window() {
        let mut lane = Lane::default();
        let t = Instant::now();
        lane.on_dispatch(false);
        lane.on_arrival(t, 2);
        lane.on_return(t + ms(40), 8, ms(40), COALESCE, LONGEST);
        // Nobody back yet: one coalesce window from the return.
        assert_eq!(
            lane.verdict(t + ms(40), None),
            Verdict::WaitUntil(t + ms(45))
        );
        // Three of the eight come back: the window restarts from them.
        lane.on_arrival(t + ms(41), 3);
        assert_eq!(
            lane.verdict(t + ms(41), None),
            Verdict::WaitUntil(t + ms(46))
        );
        assert_eq!(
            lane.verdict(t + ms(46), None),
            Verdict::Send(FlushReason::Coalesce)
        );
    }

    #[test]
    fn a_slow_trickle_of_callers_is_waited_for_no_longer_than_the_batch_wait() {
        // 64 callers answered by a 40 ms batch; one comes back every 2 ms.
        // Each arrival restarts the coalesce window, but the gather never
        // outlasts the 15 ms batch wait (half the round trip would be 20).
        let mut lane = Lane::default();
        let t = Instant::now();
        lane.on_dispatch(false);
        lane.on_return(t, 64, ms(40), COALESCE, LONGEST);
        let mut waited_until = t;
        for step in 1..=7u64 {
            lane.on_arrival(t + ms(2 * step), 1);
            match lane.verdict(t + ms(2 * step), None) {
                Verdict::WaitUntil(deadline) => waited_until = deadline,
                verdict => panic!("still gathering at {} ms, got {verdict:?}", 2 * step),
            }
        }
        assert_eq!(waited_until, t + ms(15));
        assert_eq!(
            lane.verdict(t + ms(15), None),
            Verdict::Send(FlushReason::Coalesce)
        );

        // And never longer than half the round trip, for a short forward.
        let mut lane = Lane::default();
        lane.on_dispatch(false);
        lane.on_return(t, 8, ms(6), COALESCE, LONGEST);
        lane.on_arrival(t + ms(2), 1);
        assert_eq!(lane.verdict(t + ms(2), None), Verdict::WaitUntil(t + ms(3)));
    }

    #[test]
    fn random_arrivals_turn_the_gather_off_within_three_returns() {
        let mut lane = Lane::default();
        let mut t = Instant::now();
        let mut waited = 0;
        for _ in 0..6 {
            lane.on_dispatch(false);
            // A request arrives during the forward, later than the window
            // of the return before it.
            lane.on_arrival(t + ms(4), 1);
            t += ms(6);
            lane.on_return(t, 1, ms(6), COALESCE, LONGEST);
            match lane.verdict(t, None) {
                Verdict::WaitUntil(deadline) => {
                    waited += 1;
                    // Nobody comes back inside the window.
                    t = deadline + us(1);
                    assert_eq!(lane.verdict(t, None), Verdict::Send(FlushReason::Coalesce));
                }
                verdict => assert_eq!(verdict, Verdict::Send(FlushReason::Drain)),
            }
        }
        assert_eq!(
            waited, 3,
            "three returns with no caller back end the gather"
        );
        assert!(!lane.gathers());
    }

    #[test]
    fn callers_that_start_coming_back_turn_the_gather_on_again() {
        let mut lane = Lane::default();
        let mut t = Instant::now();
        // Drive the score down with returns nobody follows.
        for _ in 0..3 {
            lane.on_dispatch(false);
            t += ms(6);
            lane.on_return(t, 4, ms(6), COALESCE, LONGEST);
            t += ms(20);
            assert!(lane.is_idle(t));
        }
        assert!(!lane.gathers());

        // A closed loop starts. The lane is not gathering, so the first
        // return sends at once, but the arrivals behind it are still
        // scored, and the next return gathers.
        lane.on_dispatch(false);
        t += ms(6);
        lane.on_arrival(t - ms(1), 3);
        lane.on_return(t, 1, ms(6), COALESCE, LONGEST);
        assert_eq!(lane.verdict(t, None), Verdict::Send(FlushReason::Drain));
        lane.on_dispatch(false);
        lane.on_arrival(t + us(300), 1);
        assert!(
            lane.gathers(),
            "one caller back inside the window is enough"
        );

        t += ms(6);
        assert_eq!(
            return_then_echo(&mut lane, t, 3, 3),
            Verdict::Send(FlushReason::Coalesce)
        );
    }

    #[test]
    fn a_closed_loop_keeps_one_group() {
        // Eight callers. The first leaves alone, as it must; from the
        // first return on, every batch holds all eight.
        let mut lane = Lane::default();
        let mut t = Instant::now();
        lane.on_arrival(t, 1);
        assert_eq!(
            lane.verdict(t, None),
            Verdict::Send(FlushReason::IdleBypass)
        );
        lane.on_dispatch(false);
        lane.on_arrival(t + us(200), 7);
        assert_eq!(lane.verdict(t + us(200), None), Verdict::Hold);

        t += ms(6);
        assert_eq!(
            return_then_echo(&mut lane, t, 1, 1),
            Verdict::Send(FlushReason::Coalesce)
        );
        for _ in 0..10 {
            lane.on_dispatch(false);
            t += ms(7);
            lane.on_return(t, 8, ms(7), COALESCE, LONGEST);
            // Seven of eight back: still waiting for the eighth.
            for caller in 0..7u64 {
                lane.on_arrival(t + us(100 * (caller + 1)), 1);
            }
            assert!(matches!(
                lane.verdict(t + us(800), None),
                Verdict::WaitUntil(_)
            ));
            lane.on_arrival(t + us(900), 1);
            assert_eq!(
                lane.verdict(t + us(900), None),
                Verdict::Send(FlushReason::Coalesce)
            );
        }
    }

    #[test]
    fn a_zero_coalesce_window_turns_the_gather_off() {
        let mut lane = Lane::default();
        let t = Instant::now();
        lane.on_dispatch(false);
        lane.on_arrival(t, 3);
        lane.on_return(t + ms(6), 1, ms(6), Duration::ZERO, LONGEST);
        assert_eq!(
            lane.verdict(t + ms(6), None),
            Verdict::Send(FlushReason::IdleBypass)
        );
    }

    #[test]
    fn a_second_batch_out_keeps_the_lane_busy_until_it_returns() {
        let mut lane = Lane::default();
        let t = Instant::now();
        lane.on_dispatch(false);
        lane.on_dispatch(false);
        assert_eq!(lane.in_flight(), 2);
        lane.on_return(t + ms(6), 12, ms(6), COALESCE, LONGEST);
        lane.on_arrival(t + ms(6), 5);
        assert_eq!(lane.verdict(t + ms(6), None), Verdict::Hold);
        lane.on_return(t + ms(12), 12, ms(6), COALESCE, LONGEST);
        assert!(matches!(
            lane.verdict(t + ms(12), None),
            Verdict::WaitUntil(_)
        ));
    }

    #[test]
    fn the_rest_of_a_cut_request_follows_its_first_part_at_once() {
        let mut lane = Lane::default();
        let t = Instant::now();
        // The first part of a request the item cap cut is out.
        lane.on_dispatch(true);
        assert_eq!(lane.verdict(t, None), Verdict::Send(FlushReason::Drain));
        // The batch that takes the last of it ends that.
        lane.on_dispatch(false);
        lane.on_arrival(t + ms(1), 1);
        assert_eq!(lane.verdict(t + ms(1), None), Verdict::Hold);
        // The parts that answered nobody open no watch.
        lane.on_return(t + ms(6), 0, ms(6), COALESCE, LONGEST);
        lane.on_return(t + ms(9), 1, ms(9), COALESCE, LONGEST);
        assert!(matches!(
            lane.verdict(t + ms(9), None),
            Verdict::WaitUntil(_)
        ));
    }

    #[test]
    fn a_coalesce_window_beyond_any_instant_leaves_the_other_limits_in_charge() {
        let mut lane = Lane::default();
        let t = Instant::now();
        lane.on_dispatch(false);
        lane.on_arrival(t, 1);
        lane.on_return(t + ms(10), 4, ms(10), Duration::MAX, LONGEST);
        assert_eq!(
            lane.verdict(t + ms(10), None),
            Verdict::WaitUntil(t + ms(15))
        );
    }

    #[test]
    fn an_unmatched_return_does_not_underflow() {
        let mut lane = Lane::default();
        lane.on_return(Instant::now(), 1, ms(6), COALESCE, LONGEST);
        assert_eq!(lane.in_flight(), 0);
    }
}
