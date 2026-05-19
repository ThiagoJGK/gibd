import React from 'react';
import './Button.css';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg' | 'massive';
  fullWidth?: boolean;
}

export function Button({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  fullWidth = false,
  className = '',
  ...props 
}: ButtonProps) {
  return (
    <button 
      className={`pill-shape btn btn-${variant} btn-size-${size} ${fullWidth ? 'btn-full' : ''} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
