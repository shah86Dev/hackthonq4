import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read Textbook
          </Link>
          <Link
            className="button button--primary button--lg margin-left--md"
            to="/docs/module1/1-introduction-to-physical-ai">
            Start Learning
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Home`}
      description="Physical AI & Humanoid Robotics Textbook - University Level Course Material">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className={clsx('col col--4')}>
                <h3>Complete University Textbook</h3>
                <p>Covering 4 modules with 16 chapters spanning the full academic year.</p>
              </div>
              <div className={clsx('col col--4')}>
                <h3>Advanced Topics</h3>
                <p>From foundational concepts to cutting-edge research in Physical AI.</p>
              </div>
              <div className={clsx('col col--4')}>
                <h3>Practical Applications</h3>
                <p>Real-world examples and lab exercises for hands-on learning.</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}