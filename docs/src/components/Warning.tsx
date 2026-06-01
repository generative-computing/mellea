import React from 'react';

interface WarningProps {
  children?: React.ReactNode;
}

export default function Warning({children}: WarningProps): React.ReactElement {
  return (
    <div className="admonition admonition-warning alert alert--warning">
      <div className="admonition-heading"><h5>warning</h5></div>
      <div className="admonition-content">{children}</div>
    </div>
  );
}
