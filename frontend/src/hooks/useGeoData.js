import { useState, useEffect } from 'react';
import { api } from '../utils/api';

/**
 * Custom hook to fetch and cache data from the GeoVizAI API.
 * @param {string} endpoint - One of: 'data', 'latest', 'summary',
 *   'predictions', 'shapRegression', 'shapClassifier', 'oof', 'baselines', 'correlation'
 */
export function useGeoData(endpoint) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const fetchers = {
      data:           api.getData,
      latest:         api.getLatest,
      summary:        api.getSummary,
      predictions:    api.getPredictions,
      shapRegression: api.getShapRegression,
      shapClassifier: api.getShapClassifier,
      oof:            api.getOOF,
      baselines:      api.getBaselines,
      correlation:    api.getCorrelation,
    };

    const fetchFn = fetchers[endpoint];
    if (!fetchFn) {
      setError(new Error(`Unknown endpoint: ${endpoint}`));
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    fetchFn()
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [endpoint]);

  return { data, loading, error };
}
