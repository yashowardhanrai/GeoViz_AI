import * as THREE from 'three';

/**
 * Creates the atmospheric shader material using Fresnel reflection.
 * Renders on a BackSide sphere with AdditiveBlending for a realistic glowing limb.
 */
export function createAtmosphereMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader: ATMOSPHERE_VERTEX,
    fragmentShader: ATMOSPHERE_FRAGMENT,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false
  });
}

// ─── Vertex Shader (View-Space) ──────────────────────────────────────────────

const ATMOSPHERE_VERTEX = /* glsl */`
  varying vec3 vNormal;
  varying vec3 vViewDir;

  void main() {
    // Normal in view-space
    vNormal = normalize(normalMatrix * normal);

    // View direction relative to camera at [0,0,0]
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewDir = normalize(-mvPosition.xyz);

    gl_Position = projectionMatrix * mvPosition;
  }
`;

// ─── Fragment Shader (Fresnel Limb Glow) ─────────────────────────────────────

const ATMOSPHERE_FRAGMENT = /* glsl */`
  varying vec3 vNormal;
  varying vec3 vViewDir;

  void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(vViewDir);

    // Fresnel scattering formula in view-space:
    // With BackSide rendering, N is facing inward, so dot(N, V) is positive at center,
    // and close to 0.0 near the edges of the sphere silhouette.
    // Edge intensity increases as normal becomes perpendicular to the view direction.
    float intensity = pow(0.65 - dot(N, V), 3.0);

    // Deep blue electric scattering
    vec4 atmosColor = vec4(0.32, 0.62, 1.0, 1.0);

    // Soft atmosphere glow factor
    gl_FragColor = atmosColor * intensity * 0.28;
  }
`;
