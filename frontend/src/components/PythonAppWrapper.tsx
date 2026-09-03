import { useEffect, useRef, useState } from 'react';

interface PythonAppWrapperProps {
  appUrl: string;
  title: string;
  aspectRatio?: string;
  onLoad?: () => void;
}

const MAX_ATTEMPTS = 3;
const LOAD_TIMEOUT_MS = 10000;

export function PythonAppWrapper({
  appUrl,
  title,
  aspectRatio = '16/9',
  onLoad,
}: PythonAppWrapperProps) {
  const [attempt, setAttempt] = useState(1);
  const [loadedRequest, setLoadedRequest] = useState<string | null>(null);
  const [failedRequest, setFailedRequest] = useState<string | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestKey = `${appUrl}-${attempt}`;

  useEffect(() => {
    timeoutRef.current = setTimeout(() => {
      setFailedRequest(requestKey);
    }, LOAD_TIMEOUT_MS);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [requestKey]);

  const handleLoad = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    setLoadedRequest(requestKey);
    onLoad?.();
  };

  const handleRetry = () => {
    if (attempt < MAX_ATTEMPTS) {
      setAttempt((currentAttempt) => currentAttempt + 1);
      return;
    }

    setAttempt(1);
  };

  const hasMoreAttempts = attempt < MAX_ATTEMPTS;
  const isLoaded = loadedRequest === requestKey;
  const hasError = failedRequest === requestKey;

  return (
    <section
      className="w-full overflow-x-hidden rounded-[2rem] border border-border-flat bg-background-base"
      aria-label={title}
    >
      <header className="border-b border-border-flat px-5 py-4 sm:px-7">
        <h2 className="text-base font-bold text-text-primary sm:text-lg">{title}</h2>
      </header>

      <div
        className="relative w-full overflow-hidden bg-black"
        style={{ aspectRatio }}
      >
        {!isLoaded && !hasError && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-background-base text-sm text-text-secondary">
            <span
              className="h-9 w-9 animate-spin rounded-full border-4 border-[#FF5500]/25 border-t-[#FF5500]"
              aria-hidden="true"
            />
            <span>Cargando aplicación...</span>
          </div>
        )}

        {hasError && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-4 bg-background-base px-6 text-center">
            <p className="max-w-md text-sm text-text-secondary sm:text-base">
              La aplicación de Python no está disponible en este momento.
            </p>
            <button
              type="button"
              onClick={handleRetry}
              className="rounded-full bg-primary-container px-5 py-2.5 text-sm font-bold text-white transition-opacity hover:opacity-85 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-container"
            >
              {hasMoreAttempts ? 'Reintentar' : 'Intentar de nuevo'}
            </button>
          </div>
        )}

        <iframe
          key={`${appUrl}-${attempt}`}
          src={appUrl}
          title={title}
          onLoad={handleLoad}
          onError={() => setFailedRequest(requestKey)}
          className={`absolute inset-0 h-full w-full border-0 ${isLoaded ? 'opacity-100' : 'opacity-0'}`}
          style={{ transition: 'opacity 180ms ease-in' }}
        />
      </div>
    </section>
  );
}