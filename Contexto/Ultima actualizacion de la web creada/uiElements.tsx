
import React from 'react';
import { UploadedImage } from './types'; // Import UploadedImage type

export const LoadingSpinner: React.FC = () => (
  <div className="flex justify-center items-center my-8">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-sky-500"></div>
    <p className="ml-3 text-sky-400 animate-text-pulse">Procesando...</p>
  </div>
);

interface ErrorMessageProps {
  message: string;
}
export const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => (
  <div className="bg-red-700 border border-red-900 text-red-100 px-4 py-3 rounded-2xl relative my-4 shadow-lg" role="alert">
    <strong className="font-bold">¡Error!</strong>
    <span className="block sm:inline ml-2">{message}</span>
  </div>
);

interface FileUploadButtonProps {
  onFileSelect?: (file: File) => void;
  onFilesSelect?: (files: File[]) => void;
  accept: string;
  label: string;
  isDirectoryUpload?: boolean;
  className?: string;
  id?: string;
  variant?: 'sky' | 'amber';
  disabled?: boolean;
}

export const FileUploadButton: React.FC<FileUploadButtonProps> = ({
  onFileSelect,
  onFilesSelect,
  accept,
  label,
  isDirectoryUpload = false,
  className = '',
  id = `file-upload-${label.replace(/\s+/g, '-').toLowerCase()}`,
  variant = 'sky',
  disabled = false,
}) => {
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      if (isDirectoryUpload && onFilesSelect) {
        onFilesSelect(Array.from(event.target.files));
      } else if (!isDirectoryUpload && event.target.files.length > 0 && onFileSelect) {
        onFileSelect(event.target.files[0]);
      }
    }
    if (inputRef.current) {
      inputRef.current.value = ""; 
    }
  };

  const handleClick = () => {
    if (!disabled) {
      inputRef.current?.click();
    }
  };

  const commonInputProps: React.InputHTMLAttributes<HTMLInputElement> = {
    type: 'file',
    onChange: handleChange,
    accept: accept,
    className: 'hidden',
    id: id,
    disabled: disabled,
  };

  const directoryUploadProps = isDirectoryUpload
    ? { webkitdirectory: "", multiple: true }
    : { multiple: false };
  
  const buttonBaseStyle = "text-white font-semibold py-3 px-6 rounded-full shadow-md transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-opacity-50";
  const variantStyle = variant === 'sky' 
    ? `bg-sky-600 hover:bg-sky-700 focus:ring-sky-500 button-glow-sky`
    : `bg-amber-600 hover:bg-amber-700 focus:ring-amber-500 button-glow-amber`;
  const disabledStyle = disabled ? "opacity-50 cursor-not-allowed" : "";

  return (
    <>
      <input
        ref={inputRef}
        {...commonInputProps}
        {...directoryUploadProps}
      />
      <button
        type="button"
        onClick={handleClick}
        className={`${buttonBaseStyle} ${variantStyle} ${className} ${disabledStyle}`}
        disabled={disabled}
        aria-disabled={disabled}
      >
        {label}
      </button>
    </>
  );
};


interface CardProps {
  children: React.ReactNode;
  className?: string;
}
export const Card: React.FC<CardProps> = ({ children, className }) => (
  <div className={`bg-slate-800 p-6 sm:p-8 shadow-2xl rounded-3xl card-hover-glow ${className}`}>
    {children}
  </div>
);

export const ImageCard: React.FC<{ image: UploadedImage; modelName: string; index: number, isQueryImage?: boolean, similarity?: number }> = 
 ({ image, modelName, index, isQueryImage = false, similarity }) => (
  <div className={`group bg-slate-700 p-2 rounded-2xl shadow-lg overflow-hidden transition-all duration-300 ease-in-out transform hover:scale-105 hover:shadow-xl ${isQueryImage ? 'border-2 border-sky-500 selected-item-glow hover:shadow-sky-500/60' : 'card-hover-glow hover:shadow-sky-500/40'}`}>
    <img
      src={`data:${image.mimeType};base64,${image.base64}`}
      alt={image.name || `${isQueryImage ? 'Imagen de consulta' : `Imagen ${index + 1} de ${modelName}`}`}
      className="w-full h-48 object-cover rounded-xl mb-2"
      loading="lazy"
    />
    <p className="text-xs text-slate-400 truncate text-center px-1" title={image.name}>
      {image.name || `${isQueryImage ? 'Imagen Consulta' : `Imagen ${index + 1}`}`}
    </p>
    {similarity !== undefined && (
        <p className="text-xs text-sky-400 text-center mt-0.5">
            Similitud (API): {similarity.toFixed(4)}
        </p>
    )}
  </div>
);

interface TabButtonProps {
  onClick: () => void;
  isActive: boolean;
  children: React.ReactNode;
  forceDisableGlow?: boolean; // New prop
}

export const TabButton: React.FC<TabButtonProps> = ({ onClick, isActive, children, forceDisableGlow = false }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 text-sm sm:px-5 sm:py-3 sm:text-base font-bold leading-normal tracking-[0.015em] transition-all duration-200 ease-in-out rounded-full
                ${isActive
                  ? `bg-sky-600 text-white shadow-xl scale-105 ${!forceDisableGlow ? 'tab-active-glow' : ''}`
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600 hover:text-white hover:button-glow-sky'}`}
    aria-pressed={isActive}
  >
    {children}
  </button>
);

interface SwitchToggleProps {
  labelLeft: string;
  labelRight: string;
  isChecked: boolean;
  onChange: (isChecked: boolean) => void;
  id?: string;
}

export const SwitchToggle: React.FC<SwitchToggleProps> = ({
  labelLeft,
  labelRight,
  isChecked,
  onChange,
  id = 'mode-switch'
}) => {
  const handleToggle = () => {
    onChange(!isChecked);
  };

  return (
    <div className="flex items-center space-x-2">
      <span className={`text-sm font-medium ${!isChecked ? 'text-sky-400' : 'text-slate-400'}`}>{labelLeft}</span>
      <button
        id={id}
        role="switch"
        aria-checked={isChecked}
        onClick={handleToggle}
        className={`relative inline-flex items-center h-7 w-14 rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-sky-500 ${
          isChecked ? 'bg-sky-600' : 'bg-slate-600'
        }`}
      >
        <span
          className={`inline-block w-6 h-6 transform bg-white rounded-full transition-transform duration-200 ease-in-out ${
            isChecked ? 'translate-x-7 shadow-lg shadow-sky-400/80' : 'translate-x-1'
          }`}
        />
      </button>
      <span className={`text-sm font-medium ${isChecked ? 'text-sky-400' : 'text-slate-400'}`}>{labelRight}</span>
    </div>
  );
};