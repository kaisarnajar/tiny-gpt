import { useState, type FormEvent } from 'react'
import './App.css'

type CompleteResponse = { sentence: string } | { error: string }

function App() {
  const [prompt, setPrompt] = useState('')
  const [sentence, setSentence] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function complete(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSentence(null)
    try {
      const res = await fetch('/api/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })
      const data: CompleteResponse = await res.json()
      if ('error' in data) setError(data.error)
      else setSentence(data.sentence)
    } catch {
      setError('Could not reach the server. Is `python server.py` running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <h1>tiny-gpt</h1>
      <p className="subtitle">Type the start of a sentence. The model finishes it.</p>

      <form className="prompt" onSubmit={complete}>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="The Dog"
          aria-label="Start of a sentence"
          autoFocus
        />
        <button type="submit" disabled={loading || !prompt.trim()}>
          {loading ? 'Thinking…' : 'Complete'}
        </button>
      </form>

      {sentence && <p className="result">{sentence}</p>}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </main>
  )
}

export default App
