import React from 'react';

interface CardGroupProps {
  cols?: number;
  children?: React.ReactNode;
}

export default function CardGroup({cols = 2, children}: CardGroupProps): React.ReactElement {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: `repeat(${cols}, 1fr)`,
      gap: '1rem',
      marginBottom: '1rem',
    }}>
      {children}
    </div>
  );
}
