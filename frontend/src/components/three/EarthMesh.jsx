import { useRef, useMemo, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { createEarthMaterial } from './shaders/earthShader';

/**
 * Renders the Earth sphere with custom day/night shaders and a rotating clouds layer.
 * Any children passed (like country highlights) will rotate in sync with the Earth.
 */
// Global module-level texture cache to prevent reloading/flickering on route navigation
let cachedTextures = null;

export default function EarthMesh({ children }) {
  const earthGroupRef = useRef();
  const cloudsRef = useRef();
  const [textures, setTextures] = useState(cachedTextures);

  // Asynchronously load NASA Blue Marble textures once
  useEffect(() => {
    if (cachedTextures) {
      // Already loaded, state initialized via useState(cachedTextures)
      return;
    }

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
      loader.load(
        urls[key],
        (tex) => {
          tex.colorSpace = THREE.SRGBColorSpace;
          texs[key] = tex;
          loaded++;
          if (loaded === keys.length) {
            cachedTextures = texs;
            setTextures(texs);
          }
        },
        undefined,
        (err) => {
          console.error('Error loading Earth texture:', key, err);
        }
      );
    });
  }, []);

  const sunDir = useMemo(() => new THREE.Vector3(3, 2, 4).normalize(), []);

  // WebGL Shader for Earth surface
  const earthMaterial = useMemo(() => {
    if (!textures) return null;
    return createEarthMaterial(textures, sunDir);
  }, [textures, sunDir]);

  // Initial rotation offset to show Eurasia/Middle East on load
  const rotationOffset = 0.7; // ~40 degrees

  useFrame(({ clock }) => {
    const elapsed = clock.getElapsedTime();
    if (earthGroupRef.current) {
      // Rotate the entire Earth group (mesh + country highlights rotate together)
      earthGroupRef.current.rotation.y = rotationOffset + elapsed * 0.025;
    }
    if (cloudsRef.current) {
      // Clouds rotate slightly faster and on a separate axis for realistic parallax
      cloudsRef.current.rotation.y = rotationOffset + elapsed * 0.033;
      cloudsRef.current.rotation.x = elapsed * 0.003;
    }
  });

  return (
    <group rotation={[0, 0, 23.44 * Math.PI / 180]}>
      {/* Group containing Earth and any conformed overlays (highlights) */}
      <group ref={earthGroupRef}>
        {textures && earthMaterial ? (
          <mesh>
            {/* High resolution geometry to prevent vertex artifacts on close-up */}
            <sphereGeometry args={[0.85, 128, 128]} />
            <primitive object={earthMaterial} attach="material" />
          </mesh>
        ) : (
          <mesh>
            {/* Fallback wireframe mesh while loading textures */}
            <sphereGeometry args={[0.85, 48, 48]} />
            <meshPhongMaterial color="#0b1229" wireframe transparent opacity={0.25} />
          </mesh>
        )}

        {/* Child highlights (e.g. CountryHighlights component) */}
        {children}
      </group>

      {/* 2. Clouds Layer */}
      {textures?.clouds && (
        <mesh ref={cloudsRef}>
          <sphereGeometry args={[0.8568, 96, 96]} />
          <meshPhongMaterial
            alphaMap={textures.clouds}
            color="#ffffff"
            transparent
            opacity={0.14}
            depthWrite={false}
            blending={THREE.NormalBlending}
          />
        </mesh>
      )}
    </group>
  );
}
