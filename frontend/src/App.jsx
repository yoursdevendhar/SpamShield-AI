import { useState } from "react";
import "./App.css";

function App(){
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function detectSpam(){
    if(!message.trim()){
      setError("Please Enter Message First!");
      setResult(null);
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try{
      const response = await fetch(
        "http://127.0.0.1:8000/api/predict",
        {
          method:"POST",
          headers:{
            "Content-Type":"application/json",
          },
          body:JSON.stringify({message}),
        }
      );
      const data = await response.json();
      if(!response.ok){
        throw new Error(data.detail||"Prediction failed");
      }
      setResult(data);
    }
      catch(err){
        setError(err.message === "Failed to fetch"
          ? "Cannot connect to the API. Check that FastAPI is running."
          : err.message
        );
      }
      finally{
        setLoading(false);
    }
  }
  return (
    <main className="App">
      <header className="header">
        <div className="logo">$</div>
        <h1>SpamShield AI</h1>
        <p>Intelligent SMS spam detection</p>
      </header>
      <section className="detector">
        <h2>Check your message</h2>
        <p className="description">
          Paste an SMS below to check whether it looks like spam.
        </p>
        <label htmlFor="sms">Your SMS/GMail Message</label>
        <textarea
          id="sms"
          value={message}
          onChange={(event)=>setMessage(event.target.value)}
          placeholder="Example: Congratulations! You won a free prize..."
          maxLength={5000}
          rows={6}
        />
        <div className="character-count">
          {message.length}/5000 characters
        </div>
        <button
          onClick={detectSpam}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Detect Spam"}
        </button>
        {error &&(
          <div className="error" role="alert">
            {error}
          </div>
        )}
        {result &&(
          <div className={`result ${result.prediction === "Spam"?"spam":"safe"}`} aria-live="polite">
            <h3>
            {result.prediction === "Spam"
              ? "⚠️ Spam Detected"
              : "✓ Not Spam"}
            </h3>
            <p>
              Estimated confidence: {result.confidence}%
            </p>
            {result.warning && (
              <p className="warning">
                {result.warning}
              </p>
            )}

            <p className="result-message">
              Message: {result.message}
            </p>
          </div>

        )}

      </section>
      <footer>
        Powered by TF-IDF and Multinomial Naive Bayes
      </footer>
    </main>
  );

}
export default App;