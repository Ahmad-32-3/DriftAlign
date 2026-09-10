import { AblationBars } from './components/story/AblationBars'
import { DomainCloud } from './components/story/DomainCloud'
import { ResultBars } from './components/story/ResultBars'
import { StackGrid } from './components/story/StackGrid'
import { StoryBeat } from './components/story/StoryBeat'
import {
  COMPARISON,
  DECISIONS,
  ILLUSTRATIVE,
  METRICS,
  NEXT,
  PROTOCOL,
  SECTORS,
} from './data'

const TOC = [
  { href: '#problem', label: 'The problem' },
  { href: '#answer', label: 'The approach' },
  { href: '#result', label: 'The result' },
  { href: '#stack', label: 'How it works' },
  { href: '#decisions', label: 'Design choices' },
  { href: '#use', label: 'Running it' },
  { href: '#next', label: 'What is next' },
]

const NOTE = ILLUSTRATIVE ? ' Placeholder numbers until the pipeline runs.' : ''

export function App() {
  return (
    <>
      <a className="skip-link" href="#problem">
        Skip to the walkthrough
      </a>

      <div className="masthead">
        <div className="masthead__inner">
          <div className="masthead__mark">
            <b>DriftAlign</b> · shifted data, then try lining the clouds up
          </div>
          <ul className="masthead__nav">
            <li><a href="#problem">problem</a></li>
            <li><a href="#answer">approach</a></li>
            <li><a href="#result">result</a></li>
            <li><a href="#stack">how</a></li>
          </ul>
        </div>
      </div>

      <main className="page">
        <header className="page-hero">
          <p className="meta">A walkthrough · train where the data looks one way, score where it looks shifted</p>
          <h1>DriftAlign</h1>
          <p className="lead">
            I train a classifier where the data looks one way, then I test where the data looks
            shifted. Before scoring, I try a step that slides the training cloud toward the test
            cloud. The question is simple: did that move raise accuracy on the shifted data, or does
            the same classifier with no transport step already win?
          </p>
          <p className="intro-detail">
            Mixing both worlds and shuffling is a debug split. The number I report is accuracy on the
            shifted data after lining the clouds up, printed next to the same classifier with no
            lining-up step.
            {NOTE}
          </p>
          <nav aria-label="On this page">
            <ul className="toc">
              {TOC.map((item) => (
                <li key={item.href}>
                  <a href={item.href}>{item.label}</a>
                </li>
              ))}
            </ul>
          </nav>
        </header>

        <StoryBeat
          id="problem"
          kicker="The problem"
          title="The labels match. The features do not."
          caption={`Circles are source. Squares are target. Same three classes, target translated in the plane. A source-trained boundary sits on empty space.${NOTE}`}
          visual={
            <>
              <DomainCloud />
              <ul className="legend">
                <li><span className="swatch" style={{ background: 'var(--accent)', borderRadius: '50%' }} /> class 0</li>
                <li><span className="swatch" style={{ background: 'var(--good)', borderRadius: '50%' }} /> class 1</li>
                <li><span className="swatch" style={{ background: 'var(--amber)', borderRadius: '50%' }} /> class 2</li>
              </ul>
            </>
          }
        >
          <p>
            I can train a classifier that looks sharp on the data it came from and still fail the
            moment the same labels show up somewhere else. Humidity, a different camera, a new
            scanner: the features slide, the labels stay. If I dump both domains into one pile and
            shuffle, the accuracy can look fine while the target is still a miss.
          </p>
          <p>
            I need a number that only counts target points. And I need a baseline that never saw
            the transport step, on that same target, so I can tell whether the alignment did
            anything.
          </p>
        </StoryBeat>

        <StoryBeat
          id="answer"
          kicker="The approach"
          title="Move the source onto the target, then fit"
          caption="I match the two unlabeled clouds, map each training point onto the test cloud, fit there, and score only on the shifted points. The matcher is a Sinkhorn step."
          visual={
            <div className="teach-card">
              <h3 className="teach-card__title">Headline split</h3>
              <dl className="stat-grid">
                <div>
                  <dt>Train</dt>
                  <dd>source only</dd>
                </div>
                <div>
                  <dt>Score</dt>
                  <dd>target only</dd>
                </div>
                <div>
                  <dt>Cost</dt>
                  <dd>{PROTOCOL.cost}</dd>
                </div>
                <div>
                  <dt>ε</dt>
                  <dd>{PROTOCOL.eps}</dd>
                </div>
              </dl>
              <p className="meta" style={{ textTransform: 'none', letterSpacing: 0, margin: 0 }}>
                {PROTOCOL.nSource} source points, {PROTOCOL.nTarget} target points, {PROTOCOL.nClasses} classes.
                IID shuffle of mixed domains is rejected as a leak.
              </p>
            </div>
          }
        >
          <p>
            Target labels stay out of training. A Sinkhorn step builds a coupling between the two
            feature clouds with a squared Euclidean cost. Each source point is replaced by a weighted
            average of target points. I fit a logistic head on those mapped points and their original
            source labels.
          </p>
          <p>
            The no-move baseline is the same head, fit on the raw source, scored on the same target.
            If lining the clouds up is doing work, the first number is higher. Majority class and chance sit
            in the table so a lucky split cannot hide.
          </p>
        </StoryBeat>

        <StoryBeat
          id="result"
          kicker="The result"
          title="Lining the clouds up holds the shifted test. Skipping that step does not."
          caption={`Accuracy on the shifted data. Lined-up ${METRICS.otPct}%, no lining-up ${METRICS.noOtPct}%, majority ${METRICS.majorityPct}%, chance ${METRICS.chancePct}%.${NOTE}`}
          visual={<ResultBars />}
        >
          <p>
            On this shift, lining the clouds up lands at {METRICS.successPct}% on the shifted data. The same
            classifier with no transport is at {METRICS.noOtPct}%. Majority is {METRICS.majorityPct}%.
            Chance is {METRICS.chancePct}%. Those figures are written by <code>scripts/run.py</code>
            into <code>data.ts</code>.
          </p>
          <dl className="stat-grid" style={{ marginTop: 'var(--space-5)' }}>
            {COMPARISON.map((row) => (
              <div key={row.label}>
                <dt>{row.label}</dt>
                <dd>{row.pct.toFixed(1)}%</dd>
              </div>
            ))}
          </dl>
          <p style={{ marginTop: 'var(--space-5)' }}>
            Ablations keep the same seed and the same source/target split. Squared Euclidean holds.
            Plain Euclidean is close. Cosine as the cost collapses: lining up falls below no lining-up.
            Sample cap at 100 still clears the floor. Green bars are lined-up; red bars are not.
          </p>
          <AblationBars />
        </StoryBeat>

        <section className="story-beat" id="stack">
          <p className="story-kicker">How it works</p>
          <h2>The tools, in plain terms</h2>
          <p className="stack-intro">
            Small stack, so the run is a clone and a Python command. Each card is what it does,
            then how.
          </p>
          <StackGrid />
        </section>

        <StoryBeat
          id="decisions"
          kicker="Design choices"
          title="The calls I made"
          caption="What I first reached for, and what I built instead."
          visual={
            <div className="teach-card">
              <h3 className="teach-card__title">First idea, and what I built</h3>
              <table className="choice-table">
                <caption className="sr-only">Design choices</caption>
                <thead>
                  <tr>
                    <th scope="col">First idea</th>
                    <th scope="col">What I built</th>
                  </tr>
                </thead>
                <tbody>
                  {DECISIONS.map((d) => (
                    <tr key={d.first}>
                      <td>{d.first}</td>
                      <td>{d.built}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          }
        >
          <p>
            <strong>I score the target, never a shuffled mix.</strong> A mixed IID accuracy can
            look high while the target is still wrong. The leak tests fail if source ids or labels
            enter that score.
          </p>
          <p>
            <strong>I kept the head small.</strong> The question is whether transport helps, not
            whether a bigger net can memorize both clouds.
          </p>
          <p>
            <strong>I measured a Gaussian shift first.</strong> MNIST to USPS is the next pair once
            those files are on disk. Until then the number is real on the synthetic shift, with the
            cap in DESIGN.
          </p>
        </StoryBeat>

        <section className="story-beat" id="use">
          <p className="story-kicker">Running it</p>
          <h2>Clone it and run it</h2>
          <ol className="stack-list" style={{ maxWidth: 'var(--measure)' }}>
            <li>From this folder, <code>python -m pip install -e .</code> then <code>python scripts/run.py</code>.</li>
            <li>It prints OT success_pct next to no-OT on the target, writes <code>metrics.json</code>, and refreshes <code>web/src/metrics.gen.ts</code>.</li>
            <li><code>npm --prefix web install</code> and <code>npm --prefix web run dev</code> to read this page. The charts pull from <code>data.ts</code>.</li>
          </ol>
          <p className="stack-intro" style={{ marginTop: 'var(--space-6)' }}>
            Where this shows up outside a demo
          </p>
          <ul className="stack-list" style={{ maxWidth: 'var(--measure)' }}>
            {SECTORS.map((s) => (
              <li key={s.name}>
                <strong>{s.name}.</strong> {s.note}
              </li>
            ))}
          </ul>
        </section>

        <section className="story-beat" id="next">
          <p className="story-kicker">What is next</p>
          <h2>Where I would take it</h2>
          <ul className="stack-list" style={{ maxWidth: 'var(--measure)' }}>
            {NEXT.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </section>

        <footer
          id="close"
          style={{
            borderTop: '1px solid var(--line-rule)',
            paddingTop: 'var(--space-6)',
            marginTop: 'var(--space-6)',
            color: 'var(--fg-low)',
            fontSize: 'var(--fs-sm)',
          }}
        >
          <p style={{ maxWidth: 'var(--measure)' }}>
            Portfolio run on a capped 2D Gaussian translation, {PROTOCOL.nSource} points per side.
            Headline is accuracy on the shifted data with the lining-up step versus without it. Not a
            production domain adapter and not an ImageNet crawl.{ILLUSTRATIVE ? ' The numbers here are placeholders until the pipeline replaces them.' : ''}
          </p>
        </footer>
      </main>
    </>
  )
}
