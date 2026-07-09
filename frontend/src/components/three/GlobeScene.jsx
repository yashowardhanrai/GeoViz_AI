import { useState, useEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import Starfield from './Starfield';
import EarthMesh from './EarthMesh';
import Atmosphere from './Atmosphere';
import CountryHighlights from './CountryHighlights';

/**
 * Diagnostic aspect fixer to prevent horizontal stretching of the globe
 * on extremely wide layouts. Automatically forces projection update.
 */
function AspectFixer() {
  const { size, camera } = useThree();
  useFrame(() => {
    const targetAspect = size.width / size.height;
    if (Math.abs(camera.aspect - targetAspect) > 0.001) {
      camera.aspect = targetAspect;
      camera.updateProjectionMatrix();
    }
  });
  return null;
}

/**
 * Orchestrator component for the 3D geopolitical risk intelligence globe.
 * Composes Starfield, EarthMesh, Atmosphere, and CountryHighlights inside
 * a canvas with cinematically tuned lighting and smooth damping camera controls.
 */
export default function GlobeScene({ height = 420 }) {
  const [geoJsonData, setGeoJsonData] = useState(null);
  const [hoveredCountry, setHoveredCountry] = useState(null);
  const [fadeOpacity, setFadeOpacity] = useState(0);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Fetch world boundaries GeoJSON asynchronously with reliable Natural Earth fallback chain
  useEffect(() => {
    const loadGeoJson = async () => {
      let data = null;
      
      // Attempt 1: Try Natural Earth 110m (rich metadata, precise borders)
      try {
        const res = await fetch('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson');
        if (res.ok) data = await res.json();
      } catch (e) { /* fallback below */ }

      // Attempt 2: Fallback datasets 110m
      if (!data) {
        try {
          const res = await fetch('https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson');
          if (res.ok) data = await res.json();
        } catch (e) {
          console.error('Failed to load world GeoJSON boundary data:', e);
        }
      }

      if (data) {
        setGeoJsonData(data);
        // Smoothly fade in overlays over 600ms
        let start = null;
        const animate = (timestamp) => {
          if (!start) start = timestamp;
          const progress = (timestamp - start) / 600;
          if (progress < 1) {
            setFadeOpacity(progress);
            requestAnimationFrame(animate);
          } else {
            setFadeOpacity(1);
          }
        };
        requestAnimationFrame(animate);
      }
    };
    loadGeoJson();
  }, []);

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setMousePos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  return (
    <div 
      className="three-canvas-container" 
      style={{ height, position: 'relative', overflow: 'hidden' }}
      onMouseMove={handleMouseMove}
    >
      {/* 3D WebGL Canvas */}
      <Canvas
        camera={{ position: [0, 0.2, 3.0], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
        style={{ background: 'transparent' }}
      >
        {/* Aspect Ratio Constraint Fixer */}
        <AspectFixer />

        {/* Cinematic Physically-Based Lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[3, 2, 4]} intensity={1.6} color="#ffffff" />
        <hemisphereLight skyColor="#b1e1ff" groundColor="#1a1a2e" intensity={0.3} />

        {/* 1. Starfield Background (Deep Space) */}
        <Starfield count={3000} radius={220} />

        {/* 2. Earth Mesh (Body + Cloud layers) + Country Highlights rotating inside */}
        <EarthMesh>
          <CountryHighlights
            geoJsonData={geoJsonData}
            onHover={setHoveredCountry}
            hoveredCountry={hoveredCountry}
            fadeOpacity={fadeOpacity}
          />
        </EarthMesh>

        {/* 3. Fresnel Atmospheric Limb Glow */}
        <Atmosphere />

        {/* Smooth Orbit Camera Controls */}
        <OrbitControls
          enableZoom={true}
          minDistance={1.6}
          maxDistance={5.0}
          enablePan={false}
          autoRotate={true}
          autoRotateSpeed={0.25}
          enableDamping={true}
          dampingFactor={0.05}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={Math.PI * 3 / 4}
        />
      </Canvas>

      {/* Floating HUD Tooltip on Hover */}
      {hoveredCountry && (
        <div style={{
          position: 'absolute',
          left: `${mousePos.x + 15}px`,
          top: `${mousePos.y + 15}px`,
          background: 'rgba(2, 6, 20, 0.95)',
          border: `1px solid ${hoveredCountry.color}`,
          borderRadius: '4px',
          padding: '6px 10px',
          color: '#E2E8F0',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '11px',
          pointerEvents: 'none',
          zIndex: 100,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.65)'
        }}>
          <div style={{ fontWeight: '800', color: hoveredCountry.color }}>
            {hoveredCountry.label.toUpperCase()}
          </div>
          <div style={{ color: '#94A3B8', fontSize: '9px', marginTop: '3px' }}>
            RISK INDEX: <span style={{ color: '#F8FAFC', fontWeight: 'bold' }}>{hoveredCountry.riskScore.toFixed(1)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
