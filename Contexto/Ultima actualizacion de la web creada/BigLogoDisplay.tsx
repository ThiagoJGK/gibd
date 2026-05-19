import React from 'react';
import Logo from './Logo';

interface BigLogoDisplayProps {
  message: string;
}

const BigLogoDisplay: React.FC<BigLogoDisplayProps> = ({ message }) => {
  return (
    <div className="flex flex-col items-center justify-center text-center h-full py-10 animate-scale-fade-in">
      <div className="w-48 h-48 mb-6 text-slate-500"> {/* Adjusted size and color */}
        <Logo />
      </div>
      <p className="text-xl text-slate-400">{message}</p>
    </div>
  );
};

export default BigLogoDisplay;