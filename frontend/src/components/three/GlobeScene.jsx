import { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import * as THREE from 'three';

// Geopolitical target countries config
const TARGET_COUNTRIES = [
  { id: 'ukraine',   label: 'Ukraine',    riskScore: 89.2, color: '#FF3040' },
  { id: 'russia',    label: 'Russia',     riskScore: 74.5, color: '#FF9100' },
  { id: 'israel',    label: 'Israel',     riskScore: 86.7, color: '#FF3040' },
  { id: 'palestine', label: 'Palestine',  riskScore: 91.0, color: '#D50000' },
  { id: 'iran',      label: 'Iran',       riskScore: 78.3, color: '#FF6F00' }
];

/**
 * Spherical mapping conversion from WGS84 coordinates to 3D Cartesian space.
 */
function latLonToVec3(lat, lon, radius = 1.0) {
  const radLat = lat * Math.PI / 180;
  const radLon = lon * Math.PI / 180;
  const x = radius * Math.cos(radLat) * Math.sin(radLon);
  const y = radius * Math.sin(radLat);
  const z = radius * Math.cos(radLat) * Math.cos(radLon);
  return new THREE.Vector3(x, y, z);
}

/**
 * Fixes antimeridian crossing (e.g. Russia's far east) by shifting negative longitudes by 360 deg
 * if the ring crosses the antimeridian, preventing Earcut triangulation from drawing lines through the globe.
 */
function adjustAntimeridian(ring) {
  let hasPos = false;
  let hasNeg = false;
  let minLon = 180;
  let maxLon = -180;
  
  for (let i = 0; i < ring.length; i++) {
    const lon = ring[i][0];
    if (lon > 0) hasPos = true;
    if (lon < 0) hasNeg = true;
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
  }
  
  if (hasPos && hasNeg && (maxLon - minLon > 180)) {
    return ring.map(pt => [pt[0] < 0 ? pt[0] + 360 : pt[0], pt[1]]);
  }
  return ring;
}

/**
 * Converts a 2D GeoJSON polygon ring coordinate loop into a curved 3D ShapeGeometry conformed to the sphere.
 */
function geoJsonToThreeGeometry(coords, radius = 1.001) {
  const shape = new THREE.Shape();
  
  // Outer ring path
  const outerRing = coords[0];
  if (outerRing.length === 0) return null;
  shape.moveTo(outerRing[0][0], outerRing[0][1]);
  for (let i = 1; i < outerRing.length; i++) {
    shape.lineTo(outerRing[i][0], outerRing[i][1]);
  }
  
  // Cut holes
  for (let h = 1; h < coords.length; h++) {
    const holeRing = coords[h];
    if (holeRing.length === 0) continue;
    const holePath = new THREE.Path();
    holePath.moveTo(holeRing[0][0], holeRing[0][1]);
    for (let i = 1; i < holeRing.length; i++) {
      holePath.lineTo(holeRing[i][0], holeRing[i][1]);
    }
    shape.holes.push(holePath);
  }
  
  // Triangulate shape in 2D
  const geom = new THREE.ShapeGeometry(shape);
  
  // Map vertices to 3D Cartesian coordinates on sphere
  const posAttr = geom.getAttribute('position');
  for (let i = 0; i < posAttr.count; i++) {
    const lon = posAttr.getX(i);
    const lat = posAttr.getY(i);
    const vec = latLonToVec3(lat, lon, radius);
    posAttr.setXYZ(i, vec.x, vec.y, vec.z);
  }
  geom.computeVertexNormals();
  return geom;
}

/**
 * Parses GeoJSON features into structured conformed geometries and border line paths.
 */
function getCountryGeometries(geometry, radius = 1.001) {
  const geometries = [];
  const borders = [];

  const addPolygon = (coords) => {
    const adjustedCoords = coords.map(ring => adjustAntimeridian(ring));
    const geom = geoJsonToThreeGeometry(adjustedCoords, radius);
    if (geom) geometries.push(geom);
    
    // Draw outer boundary line ring
    const borderPoints = adjustedCoords[0].map(pt => latLonToVec3(pt[1], pt[0], radius + 0.001));
    borders.push(borderPoints);
  };

  if (geometry.type === 'Polygon') {
    addPolygon(geometry.coordinates);
  } else if (geometry.type === 'MultiPolygon') {
    geometry.coordinates.forEach(polyCoords => addPolygon(polyCoords));
  }

  return { geometries, borders };
}

function CountryHighlights({ geoJsonData, onHover, hoveredCountry, fadeOpacity }) {
  // Extract and parse targets
  const countryMeshes = useMemo(() => {
    if (!geoJsonData) return [];
    
    const items = [];
    const features = geoJsonData.features || [];

    TARGET_COUNTRIES.forEach((c) => {
      // Find matching GeoJSON feature
      const feature = features.find(f => {
        const props = f.properties || {};
        const name = (props.admin || props.name || '').toLowerCase();
        const iso3 = (props.adm0_a3 || props.sov_a3 || props['iso-a3'] || '').toLowerCase();
        
        if (c.id === 'russia') return name === 'russia' || iso3 === 'rus';
        if (c.id === 'ukraine') return name === 'ukraine' || iso3 === 'ukr';
        if (c.id === 'israel') return name === 'israel' || iso3 === 'isr';
        if (c.id === 'palestine') return name === 'palestine' || name.includes('palestine') || iso3 === 'pse';
        if (c.id === 'iran') return name === 'iran' || name.includes('iran') || iso3 === 'irn';
        return false;
      });

      if (feature && feature.geometry) {
        const { geometries, borders } = getCountryGeometries(feature.geometry, 1.001);
        items.push({
          ...c,
          geometries,
          borders
        });
      }
    });

    return items;
  }, [geoJsonData]);

  return (
    <group>
      {countryMeshes.map((c) => (
        <CountryHighlightItem
          key={c.id}
          country={c}
          onHover={onHover}
          isHovered={hoveredCountry?.id === c.id}
          fadeOpacity={fadeOpacity}
        />
      ))}
    </group>
  );
}

function CountryHighlightItem({ country, onHover, isHovered, fadeOpacity }) {
  return (
    <group
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(country);
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={() => {
        onHover(null);
        document.body.style.cursor = 'default';
      }}
    >
      {/* 1. Curved overlay meshes covering the country region */}
      {country.geometries.map((geom, idx) => (
        <mesh key={idx} geometry={geom}>
          <meshBasicMaterial
            color={country.color}
            transparent
            opacity={fadeOpacity * (isHovered ? 0.32 : 0.18)}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}

      {/* 2. Glow and crisp border outlines */}
      {country.borders.map((points, idx) => (
        <group key={idx}>
          {/* Soft border outer glow (thicker, low opacity) */}
          <Line
            points={points}
            color={country.color}
            lineWidth={3.2}
            transparent
            opacity={fadeOpacity * (isHovered ? 0.5 : 0.28)}
          />
          {/* Crisp border inner line (slender, high contrast) */}
          <Line
            points={points}
            color={country.color}
            lineWidth={1.2}
            transparent
            opacity={fadeOpacity * (isHovered ? 0.9 : 0.7)}
          />
        </group>
      ))}
    </group>
  );
}

function Globe({ geoJsonData, onHover, hoveredCountry, fadeOpacity }) {
  const earthRef = useRef();
  const cloudsRef = useRef();
  const [textures, setTextures] = useState(null);

  // Asynchronously load NASA Blue Marble textures
  useEffect(() => {
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin('anonymous');

    const urls = {
      day: 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
      night: 'https://unpkg.com/three-globe/example/img/earth-night.jpg',
      bump: 'https://unpkg.com/three-globe/example/img/earth-topology.png',
      specular: 'https://unpkg.com/three-globe/example/img/earth-water.png',
      clouds: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_clouds_1024.png'
    };

    let loaded = 0;
    const texs = {};
    const keys = Object.keys(urls);

    keys.forEach(key => {
      loader.load(urls[key], (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        texs[key] = tex;
        loaded++;
        if (loaded === keys.length) {
          setTextures(texs);
        }
      }, undefined, (err) => {
        console.error('Error loading texture:', key, err);
      });
    });
  }, []);

  const sunDir = useMemo(() => new THREE.Vector3(5, 3, 5).normalize(), []);

  // WebGL Shader for blending day satellite images and night city lights
  const earthShaderMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        uDayTex: { value: null },
        uNightTex: { value: null },
        uBumpTex: { value: null },
        uSpecularTex: { value: null },
        uSunDir: { value: sunDir }
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vViewDir;
        varying vec3 vSunDir;
        uniform vec3 uSunDir;
        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vViewDir = normalize(-mvPosition.xyz);
          vSunDir = normalize(viewMatrix * vec4(uSunDir, 0.0)).xyz;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform sampler2D uDayTex;
        uniform sampler2D uNightTex;
        uniform sampler2D uBumpTex;
        uniform sampler2D uSpecularTex;
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vViewDir;
        varying vec3 vSunDir;
        void main() {
          vec3 normal = normalize(vNormal);
          vec3 viewDir = normalize(vViewDir);
          vec3 sunDir = normalize(vSunDir);
          
          vec4 dayCol = texture2D(uDayTex, vUv);
          vec4 nightCol = texture2D(uNightTex, vUv);
          vec4 specCol = texture2D(uSpecularTex, vUv);
          
          float cosTheta = dot(normal, sunDir);
          float dayMix = clamp(cosTheta * 3.0, 0.0, 1.0);
          
          vec3 reflectDir = reflect(-sunDir, normal);
          float specStrength = specCol.r;
          float specAmount = pow(max(dot(reflectDir, viewDir), 0.0), 24.0) * specStrength * 0.8;
          
          vec3 diffuse = mix(nightCol.rgb * 1.5, dayCol.rgb + vec3(specAmount), dayMix);
          diffuse += vec3(0.02) * dayCol.rgb; // Ambient fill
          
          gl_FragColor = vec4(diffuse, 1.0);
        }
      `
    });
  }, [textures, sunDir]);

  useEffect(() => {
    if (textures) {
      earthShaderMaterial.uniforms.uDayTex.value = textures.day;
      earthShaderMaterial.uniforms.uNightTex.value = textures.night;
      earthShaderMaterial.uniforms.uBumpTex.value = textures.bump;
      earthShaderMaterial.uniforms.uSpecularTex.value = textures.specular;
    }
  }, [textures, earthShaderMaterial]);

  // Atmospheric limb glow
  const atmosphereMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vViewDir = normalize(-mvPosition.xyz);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          float intensity = pow(0.7 - dot(vNormal, vViewDir), 3.0);
          gl_FragColor = vec4(0.35, 0.6, 1.0, 1.0) * intensity * 0.22;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false
    });
  }, []);

  useFrame(({ clock }) => {
    if (earthRef.current) {
      earthRef.current.rotation.y = clock.getElapsedTime() * 0.035;
    }
    if (cloudsRef.current) {
      cloudsRef.current.rotation.y = clock.getElapsedTime() * 0.043;
      cloudsRef.current.rotation.x = clock.getElapsedTime() * 0.004;
    }
  });

  return (
    <group rotation={[0, 0, 23.44 * Math.PI / 180]}>
      {/* 1. Earth Body Mesh */}
      {textures ? (
        <mesh ref={earthRef}>
          <sphereGeometry args={[1, 64, 64]} />
          <primitive object={earthShaderMaterial} attach="material" />
          
          {/* Conformed GeoJSON country highlights */}
          <CountryHighlights
            geoJsonData={geoJsonData}
            onHover={onHover}
            hoveredCountry={hoveredCountry}
            fadeOpacity={fadeOpacity}
          />
        </mesh>
      ) : (
        <mesh ref={earthRef}>
          <sphereGeometry args={[1, 48, 48]} />
          <meshPhongMaterial color="#0b1229" wireframe transparent opacity={0.25} />
        </mesh>
      )}

      {/* 2. Clouds Layer */}
      {textures?.clouds && (
        <mesh ref={cloudsRef}>
          <sphereGeometry args={[1.012, 64, 64]} />
          <meshPhongMaterial
            alphaMap={textures.clouds}
            color="#ffffff"
            transparent
            opacity={0.16}
            depthWrite={false}
            blending={THREE.NormalBlending}
          />
        </mesh>
      )}

      {/* 3. Fresnel Atmosphere limb glow */}
      <mesh>
        <sphereGeometry args={[1.13, 32, 32]} />
        <primitive object={atmosphereMaterial} attach="material" />
      </mesh>
    </group>
  );
}

export default function GlobeScene({ height = 420 }) {
  const [geoJsonData, setGeoJsonData] = useState(null);
  const [hoveredCountry, setHoveredCountry] = useState(null);
  const [fadeOpacity, setFadeOpacity] = useState(0);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Fetch world boundaries GeoJSON asynchronously
  useEffect(() => {
    fetch('https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson')
      .then(res => res.json())
      .then(data => {
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
      })
      .catch(err => console.error('Failed to load GeoJSON:', err));
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
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 0, 2.5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={0.35} />
        <directionalLight position={[5, 3, 5]} intensity={1.2} color="#ffffff" />
        
        <Globe 
          geoJsonData={geoJsonData}
          onHover={setHoveredCountry}
          hoveredCountry={hoveredCountry}
          fadeOpacity={fadeOpacity}
        />
        
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.3}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={Math.PI * 3 / 4}
        />
      </Canvas>

      {/* Cursor-Following HUD Tooltip (Only shown on hover) */}
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
