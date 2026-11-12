import '@fontsource-variable/schibsted-grotesk'
import '@fontsource/ibm-plex-mono/400.css'
import './index.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
