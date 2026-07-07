export default function LoadingState({ message = 'Loading data...' }) {
  return (
    <div className="loading-container">
      <div className="loading-spinner" />
      <div className="loading-text">{message}</div>
    </div>
  );
}
