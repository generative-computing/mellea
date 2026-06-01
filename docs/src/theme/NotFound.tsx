import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

export default function NotFound(): React.ReactElement {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="Page not found">
      <main style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh',
        padding: '2rem',
        textAlign: 'center',
      }}>
        <h1 style={{fontSize: '4rem', margin: '0 0 0.5rem'}}>404</h1>
        <p style={{fontSize: '1.25rem', marginBottom: '2rem'}}>
          Page not found — this URL doesn&apos;t exist on {siteConfig.title} docs.
        </p>
        <p style={{color: 'var(--ifm-color-emphasis-600)', marginBottom: '2rem'}}>
          The page may have moved. Try searching, or start from the home page.
        </p>
        <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center'}}>
          <Link
            to="/"
            className="button button--primary button--lg">
            Go to docs home
          </Link>
          <Link
            href="https://github.com/generative-computing/mellea/issues/new?labels=documentation&template=bug_report.md&title=Broken+link"
            className="button button--secondary button--lg">
            Report a broken link
          </Link>
        </div>
      </main>
    </Layout>
  );
}
