import React from 'react';

interface CardProps {
  title?: string;
  icon?: string;
  href?: string;
  children?: React.ReactNode;
}

export default function Card({title, href, children}: CardProps): JSX.Element {
  const content = (
    <div style={{
      border: '1px solid var(--ifm-color-emphasis-300)',
      borderRadius: '8px',
      padding: '1rem',
      marginBottom: '0.5rem',
      display: 'block',
      color: 'inherit',
    }}>
      {title && <strong style={{display: 'block', marginBottom: '0.5rem'}}>{title}</strong>}
      {children}
    </div>
  );

  if (href) {
    return <a href={href} style={{textDecoration: 'none'}}>{content}</a>;
  }
  return content;
}
