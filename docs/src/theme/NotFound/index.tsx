import React from 'react';
import Layout from '@theme/Layout';
import NotFoundContent from '@theme/NotFound/Content';

export default function NotFound(): React.ReactElement {
  return (
    <Layout title="Page not found">
      <NotFoundContent />
    </Layout>
  );
}
