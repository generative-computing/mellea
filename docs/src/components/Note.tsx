import React from 'react';

interface NoteProps {
  children?: React.ReactNode;
}

export default function Note({children}: NoteProps): React.ReactElement {
  return (
    <div className="admonition admonition-note alert alert--secondary">
      <div className="admonition-heading"><h5>note</h5></div>
      <div className="admonition-content">{children}</div>
    </div>
  );
}
