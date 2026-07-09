import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import { getCountryGeometries } from './utils/geoProjection';

// Target countries configurations matching the main application states
const TARGET_COUNTRIES = [
  { id: 'ukraine',   label: 'Ukraine',    riskScore: 89.2, color: '#FF3040' },
  { id: 'russia',    label: 'Russia',     riskScore: 74.5, color: '#FF9100' },
  { id: 'israel',    label: 'Israel',     riskScore: 86.7, color: '#FF3040' },
  { id: 'palestine', label: 'Palestine',  riskScore: 91.0, color: '#D50000' },
  { id: 'iran',      label: 'Iran',       riskScore: 78.3, color: '#FF6F00' }
];

export default function CountryHighlights({ geoJsonData, onHover, hoveredCountry, fadeOpacity }) {
  // Extract and construct Three.js meshes and border line paths for our target countries
  const countryMeshes = useMemo(() => {
    if (!geoJsonData) return [];

    const items = [];
    const features = geoJsonData.features || [];

    TARGET_COUNTRIES.forEach((c) => {
      // Find matching GeoJSON feature with case-insensitive property name checks
      const feature = features.find(f => {
        const props = f.properties || {};
        const name = (props.ADMIN || props.admin || props.NAME || props.name || '').toLowerCase();
        const iso3 = (props.ADM0_A3 || props.adm0_a3 || props.SOV_A3 || props.sov_a3 || props.ISO_A3 || props.iso_a3 || props.ISO_A3_EH || '').toLowerCase();

        if (c.id === 'russia') return name === 'russia' || iso3 === 'rus';
        if (c.id === 'ukraine') return name === 'ukraine' || iso3 === 'ukr';
        if (c.id === 'israel') return name === 'israel' || iso3 === 'isr';
        if (c.id === 'palestine') {
          return name === 'palestine' || name.includes('palestine') || name.includes('west bank') || name.includes('gaza') || iso3 === 'pse';
        }
        if (c.id === 'iran') return name === 'iran' || name.includes('iran') || iso3 === 'irn';
        return false;
      });

      if (feature && feature.geometry) {
        // Generate conformed geometries and outer boundaries at radius 1.001 (slightly above surface)
        const { geometries, borders } = getCountryGeometries(feature.geometry, 0.85085);
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
  // Compute basic material settings
  const color = useMemo(() => new THREE.Color(country.color), [country.color]);
  const fillOpacity = isHovered ? 0.32 * fadeOpacity : 0.18 * fadeOpacity;

  // Interactivity handlers
  const handlePointerOver = (e) => {
    e.stopPropagation();
    document.body.style.cursor = 'pointer';
    onHover(country);
  };

  const handlePointerOut = (e) => {
    e.stopPropagation();
    document.body.style.cursor = 'default';
    onHover(null);
  };

  return (
    <group>
      {/* 1. Translucent Surface Meshes (Fill) */}
      {country.geometries.map((geom, idx) => (
        <mesh
          key={`fill-${country.id}-${idx}`}
          geometry={geom}
          onPointerOver={handlePointerOver}
          onPointerOut={handlePointerOut}
        >
          <meshBasicMaterial
            color={color}
            transparent={true}
            opacity={fillOpacity}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}

      {/* 2. Premium Dual-Line Boundaries (Crisp inner + soft outer glow) */}
      {country.borders.map((points, idx) => {
        if (points.length < 2) return null;
        return (
          <group key={`borders-${country.id}-${idx}`}>
            {/* Soft outer boundary glow */}
            <Line
              points={points}
              color={country.color}
              lineWidth={2.8}
              transparent={true}
              opacity={0.25 * fadeOpacity}
              blending={THREE.AdditiveBlending}
            />
            {/* Crisp inner boundary line */}
            <Line
              points={points}
              color={country.color}
              lineWidth={1.0}
              transparent={true}
              opacity={0.7 * fadeOpacity}
            />
          </group>
        );
      })}
    </group>
  );
}
