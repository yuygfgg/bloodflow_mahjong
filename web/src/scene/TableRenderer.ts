import {
  AmbientLight,
  BufferGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshPhongMaterial,
  PCFSoftShadowMap,
  PerspectiveCamera,
  PlaneGeometry,
  PointLight,
  Raycaster,
  Scene,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { AnimationHint } from "../../../engine/wasm/js/src/protocol";
import type { UiSnapshot } from "../../../engine/wasm/js/src/types";
import {
  Action,
  EventKind,
  MeldKind,
  Phase,
  SEAT_NAMES,
  TILE_KIND_COUNT,
  exchangeDirectionLabel,
  expandHistogram,
  legalDiscardTiles,
  phaseTitle,
  tileSuit,
} from "../game/domain";

const TILE = { x: 0.525954, y: 0.373452, z: 0.678642 };
const HORIZONTAL_FOV = 80;
const INTRO_SECONDS = 3;
const TILE_ATLAS_STEP = { x: 0.125, y: 0.25 };
const TILE_BACK_CELL = 28;
const ASSET_ROOT = `${import.meta.env.BASE_URL}assets/`;
const WALL_SIDE_STACKS = [14, 14, 13, 13] as const;

interface WallSlot {
  side: number;
  stack: number;
  layer: number;
}

const WALL_SLOTS = buildWallSlots();
const WALL_DRAW_ORDER = buildWallDrawOrder(WALL_SLOTS);

type TileState =
  | "normal"
  | "selected"
  | "disabled"
  | "locked"
  | "winning"
  | "latest"
  | "missing"
  | "hint";

interface TileObject {
  mesh: Mesh;
  target: Vector3;
  targetRotation: Vector3;
  state: TileState;
  tile: number;
}

interface TransientTile {
  mesh: Mesh;
  sourceKey?: string;
  hideSource: boolean;
  from: Vector3;
  to: Vector3;
  fromRotation: Vector3;
  toRotation: Vector3;
  startedAt: number;
  duration: number;
  targetKey: string;
}

interface SeatTileContext {
  seat: number;
  tile: number;
  zone: "hand" | "root" | "win" | "locked" | "meld" | "river";
  index: number;
  /** Visual index after reserving a separate slot for the drawn tile. */
  layoutIndex?: number;
  state: TileState;
  action?: number;
  drawn?: boolean;
  meldKind?: number;
  meldCopy?: number;
  meldOffset?: number;
  sourceRelative?: number;
}

// Keep the three concealed/public rows physically separated. The tile model
// is scaled to TILE dimensions, so each row needs more than one full tile
// depth of clearance to avoid depth-sorting artifacts in perspective view.
const HAND_Z = 8;
const MELD_Z = 7;
const LOCKED_Z = 6.3;
const MELD_START_X = -5.2;
const WIN_Z = (HAND_Z + MELD_Z) / 2;
const DRAWN_GAP = TILE.x * 0.35;
const WIN_GAP = TILE.x * 0.9;

export interface TableRendererOptions {
  onTileClick: (tile: number, sourceKey?: string) => void;
  onReady: () => void;
  reducedMotion: boolean;
}

export class TableRenderer {
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(50, 16 / 9, 1, 1000);
  readonly renderer: WebGLRenderer;

  private readonly canvas: HTMLCanvasElement;
  private readonly loader = new GLTFLoader();
  private readonly textureLoader = new TextureLoader();
  private readonly root = new Group();
  private readonly dynamic = new Group();
  private readonly tiles = new Map<string, TileObject>();
  private readonly transientTiles = new Map<string, TransientTile>();
  private readonly geometries = new Map<number, BufferGeometry>();
  private readonly materials = new Map<TileState, MeshPhongMaterial>();
  private readonly raycaster = new Raycaster();
  private readonly pointer = new Vector2();
  private readonly options: TableRendererOptions;
  private readonly centerStatusCanvas: HTMLCanvasElement;
  private readonly centerStatusContext: CanvasRenderingContext2D;
  private readonly centerStatusTexture: CanvasTexture;
  private readonly centerStatusMesh: Mesh;
  private readonly lookTarget = new Vector3(0, -4, 0);
  private readonly currentLookTarget = new Vector3(0, 2, 0);
  private readonly cameraStart = new Vector3(0, 2, 4);
  private readonly cameraEnd = new Vector3(0, 16, 10);
  private readonly introStarted = performance.now();
  private readonly atlasPromise: Promise<Texture>;
  private atlas: Texture | undefined;
  private snapshot: UiSnapshot | null = null;
  private hintAction: number | null = null;
  private pendingExchangeSelectionKeys: readonly string[] = [];
  private hiddenRiverKeys = new Set<string>();
  private reducedMotion: boolean;
  private animationEnd = 0;
  private animationDuration = 0;
  private wallVisible = new Set<number>();
  private wallRemaining = -1;
  private animationFrame = 0;
  private transientSequence = 0;
  private hoverKey: string | null = null;
  private pressedKey: string | null = null;
  private disposed = false;

  constructor(canvas: HTMLCanvasElement, options: TableRendererOptions) {
    this.canvas = canvas;
    this.options = options;
    this.reducedMotion = options.reducedMotion;
    this.centerStatusCanvas = document.createElement("canvas");
    this.centerStatusCanvas.width = 1400;
    this.centerStatusCanvas.height = 400;
    const centerStatusContext = this.centerStatusCanvas.getContext("2d");
    if (centerStatusContext == null) {
      throw new Error("The browser does not provide a 2D canvas context.");
    }
    this.centerStatusContext = centerStatusContext;
    this.centerStatusTexture = new CanvasTexture(this.centerStatusCanvas);
    this.centerStatusTexture.colorSpace = SRGBColorSpace;
    this.centerStatusMesh = new Mesh(
      new PlaneGeometry(3.15, 0.9),
      new MeshBasicMaterial({
        map: this.centerStatusTexture,
        transparent: true,
        depthWrite: false,
        side: DoubleSide,
      }),
    );
    this.centerStatusMesh.rotation.x = -Math.PI / 2;
    // Keep text level for the viewer while the square plate below is a diamond.
    this.centerStatusMesh.position.set(0, 0.22, 0);
    this.centerStatusMesh.renderOrder = 5;
    this.centerStatusMesh.visible = false;
    this.renderer = new WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFSoftShadowMap;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene.background = new Color("#000305");
    this.scene.add(this.root);
    this.root.add(this.dynamic);
    this.root.add(this.centerStatusMesh);
    this.addLights();
    this.atlasPromise = this.loadAtlas();
    this.bindInput();
    window.addEventListener("resize", this.resize);
    this.resize();
    this.animationFrame = requestAnimationFrame(this.render);
    void this.loadStaticScene();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.animationFrame);
    window.removeEventListener("resize", this.resize);
    this.unbindInput();
    this.clearTransientTiles();
    this.renderer.dispose();
    this.centerStatusMesh.geometry.dispose();
    (this.centerStatusMesh.material as MeshBasicMaterial).dispose();
    this.centerStatusTexture.dispose();
    for (const material of this.materials.values()) material.dispose();
  }

  setSnapshot(
    snapshot: UiSnapshot | null,
    hintAction: number | null,
    pendingExchangeSelectionKeys: readonly string[] = [],
    animationHints: readonly AnimationHint[] = [],
  ): void {
    const previous = this.snapshot;
    const startsNewWall =
      snapshot != null &&
      snapshot.phase === 0 &&
      snapshot.exchangeSelectedCount === 0 &&
      (previous == null ||
        previous.phase !== 0 ||
        snapshot.eventHistory.length < previous.eventHistory.length);
    if (startsNewWall) {
      this.wallVisible.clear();
      this.wallRemaining = -1;
      this.clearTransientTiles();
    }
    this.snapshot = snapshot;
    this.updateCenterStatus();
    this.hintAction = hintAction;
    this.pendingExchangeSelectionKeys = pendingExchangeSelectionKeys;
    if (animationHints.length > 0 && !this.reducedMotion) {
      this.animationDuration = animationHints.reduce(
        (total, hint) => total + animationDurationForHint(hint),
        0,
      );
      this.animationEnd = performance.now() + this.animationDuration;
    }
    this.reconcile(animationHints, previous);
  }

  setReducedMotion(value: boolean): void {
    this.reducedMotion = value;
    if (value) this.animationEnd = 0;
  }

  private async loadAtlas(): Promise<Texture> {
    const atlas = await this.textureLoader.loadAsync(
      `${ASSET_ROOT}textures/tiles.webp`,
    );
    atlas.flipY = false;
    atlas.colorSpace = SRGBColorSpace;
    atlas.anisotropy = Math.min(
      this.renderer.capabilities.getMaxAnisotropy(),
      8,
    );
    return atlas;
  }

  private async loadStaticScene(): Promise<void> {
    try {
      const [atlas, table, field, center, tile] = await Promise.all([
        this.atlasPromise,
        this.loader.loadAsync(`${ASSET_ROOT}models/table_high.glb`),
        this.loader.loadAsync(`${ASSET_ROOT}models/field.glb`),
        this.loader.loadAsync(`${ASSET_ROOT}models/table_center.glb`),
        this.loader.loadAsync(`${ASSET_ROOT}models/tile_high.glb`),
      ]);
      this.atlas = atlas;

      const tableMesh = this.firstMesh(table.scene);
      tableMesh.material = new MeshPhongMaterial({
        map: await this.loadTexture("textures/table_high.webp"),
        shininess: 96,
      });
      tableMesh.position.y = -0.163;
      tableMesh.scale.setScalar(10);
      tableMesh.receiveShadow = true;
      this.root.add(table.scene);

      const fieldMesh = this.firstMesh(field.scene);
      fieldMesh.material = new MeshPhongMaterial({
        map: await this.loadTexture("textures/field_high.webp"),
        shininess: 0,
      });
      fieldMesh.scale.set(9.6, 1, 9.6);
      fieldMesh.position.y = -0.002;
      fieldMesh.receiveShadow = true;
      this.root.add(field.scene);

      const centerMesh = this.firstMesh(center.scene);
      centerMesh.material = new MeshPhongMaterial({
        map: await this.loadTexture("textures/table_center.webp"),
        shininess: 30,
      });
      centerMesh.scale.setScalar(TILE.x * 2.9);
      centerMesh.position.y = 0.02;
      centerMesh.receiveShadow = true;
      // Turn the square center plate into a diamond so each seat faces a corner.
      center.scene.rotation.y = Math.PI / 4;
      this.root.add(center.scene);

      const sourceMesh = this.firstMesh(tile.scene);
      const sourceGeometry = sourceMesh.geometry as BufferGeometry;
      for (let cell = 0; cell <= TILE_BACK_CELL; cell += 1) {
        this.geometries.set(cell, this.geometryForCell(sourceGeometry, cell));
      }
      // Make the tile back available before the first snapshot arrives.
      this.materials.set("normal", this.materialFor("normal", atlas));
      this.options.onReady();
      this.reconcile();
    } catch (error) {
      // The DOM shell remains usable if a low-end browser cannot load WebGL.
      console.error("Failed to load the OpenRiichi scene assets", error);
      this.options.onReady();
    }
  }

  private async loadTexture(path: string): Promise<Texture> {
    const texture = await this.textureLoader.loadAsync(`${ASSET_ROOT}${path}`);
    texture.colorSpace = SRGBColorSpace;
    texture.anisotropy = Math.min(
      this.renderer.capabilities.getMaxAnisotropy(),
      8,
    );
    return texture;
  }

  private firstMesh(root: Group): Mesh {
    let found: Mesh | undefined;
    root.traverse((object) => {
      if (found == null && object instanceof Mesh) found = object;
    });
    if (found == null) throw new Error("GLB asset does not contain a mesh.");
    return found;
  }

  private geometryForCell(
    source: BufferGeometry,
    cell: number,
  ): BufferGeometry {
    const geometry = source.clone();
    const uv = geometry.getAttribute("uv");
    const face =
      geometry.getAttribute("_face") ?? geometry.getAttribute("_FACE");
    if (uv == null || face == null) return geometry;
    const offsetX = (cell % 8) * TILE_ATLAS_STEP.x;
    const offsetY = Math.floor(cell / 8) * TILE_ATLAS_STEP.y;
    for (let index = 0; index < uv.count; index += 1) {
      if (face.getX(index) > 0.5) {
        uv.setXY(index, uv.getX(index) + offsetX, uv.getY(index) + offsetY);
      }
    }
    uv.needsUpdate = true;
    return geometry;
  }

  private materialFor(state: TileState, atlas: Texture): MeshPhongMaterial {
    const cached = this.materials.get(state);
    if (cached != null) return cached;
    const material = new MeshPhongMaterial({
      map: atlas,
      color: 0xffffff,
      shininess: 96,
      specular: 0xffffff,
    });
    switch (state) {
      case "selected":
        material.emissive = new Color("#d4a32c");
        material.emissiveIntensity = 0.72;
        break;
      case "locked":
        // The stable winning base remains in the hand as a visibly inert,
        // grey tile. It is not an extra exposed meld row.
        material.color = new Color("#92979a");
        material.emissive = new Color("#34393b");
        material.emissiveIntensity = 0.32;
        break;
      case "winning":
        material.emissive = new Color("#f1b92f");
        material.emissiveIntensity = 1.05;
        break;
      case "latest":
        material.emissive = new Color("#35b9d5");
        material.emissiveIntensity = 0.86;
        break;
      case "missing":
        material.emissive = new Color("#8b2b18");
        material.emissiveIntensity = 0.62;
        break;
      case "hint":
        material.emissive = new Color("#208f68");
        material.emissiveIntensity = 0.58;
        break;
      case "disabled":
        material.color = new Color("#777777");
        material.transparent = true;
        material.opacity = 0.45;
        break;
      default:
        break;
    }
    this.materials.set(state, material);
    return material;
  }

  /** Draw the center status onto the physical table-center plate. */
  private updateCenterStatus(): void {
    const snapshot = this.snapshot;
    const context = this.centerStatusContext;
    const width = this.centerStatusCanvas.width;
    const height = this.centerStatusCanvas.height;
    context.clearRect(0, 0, width, height);
    if (snapshot == null) {
      this.centerStatusMesh.visible = false;
      this.centerStatusTexture.needsUpdate = true;
      return;
    }

    const details: string[] = [];
    if (snapshot.phase === Phase.Exchange) {
      details.push(
        `方向 ${exchangeDirectionLabel(snapshot.exchangeDirection)}`,
      );
    }
    if (snapshot.pendingSource >= 0) {
      details.push(`来源 ${SEAT_NAMES[snapshot.pendingSource]}`);
    }

    context.save();
    context.fillStyle = "rgba(0, 5, 8, 0.76)";
    context.strokeStyle = "rgba(255, 224, 116, 0.72)";
    context.lineWidth = 5;
    context.beginPath();
    context.roundRect(24, 20, width - 48, height - 40, 30);
    context.fill();
    context.stroke();
    context.textBaseline = "middle";

    const remainingLabel = "余";
    const remainingValue = String(snapshot.wallRemaining);
    const remainingGap = 32;
    context.font = '700 172px "Bloodflow UI", "Noto Sans CJK SC", sans-serif';
    const labelWidth = context.measureText(remainingLabel).width;
    const valueWidth = context.measureText(remainingValue).width;
    let remainingX = (width - labelWidth - remainingGap - valueWidth) / 2;
    context.textAlign = "left";
    context.fillStyle = "#e9f3f7";
    context.fillText(remainingLabel, remainingX, 132);
    remainingX += labelWidth + remainingGap;
    context.fillStyle = "#ffe36c";
    context.fillText(remainingValue, remainingX, 132);

    context.textAlign = "center";
    context.fillStyle = "#f2f7fb";
    context.font = '700 52px "Bloodflow UI", "Noto Sans CJK SC", sans-serif';
    context.fillText(phaseTitle(snapshot), width / 2, 270);
    if (details.length > 0) {
      context.fillStyle = "#a9c8d8";
      context.font = '400 30px "Bloodflow UI", "Noto Sans CJK SC", sans-serif';
      context.fillText(details.join("   "), width / 2, 338);
    }
    context.restore();
    this.centerStatusMesh.visible = true;
    this.centerStatusTexture.needsUpdate = true;
  }

  private reconcile(
    animationHints: readonly AnimationHint[] = [],
    previous: UiSnapshot | null = this.snapshot,
  ): void {
    if (this.snapshot == null || this.geometries.size === 0) return;
    const next = new Map<string, SeatTileContext>();
    this.reconcileWall(next);
    for (let seat = 0; seat < 4; seat += 1) {
      this.reconcileSeat(seat, next);
    }
    const snapPersistentTiles = animationHints.some(
      (hint) => hint.kind === "meld" || hint.kind === "hu",
    );
    if (previous != null && !this.reducedMotion) {
      const usedHandSources = new Set<string>();
      this.createDiscardAnimations(
        previous,
        next,
        animationHints,
        usedHandSources,
      );
      this.createWinAnimations(previous, next, animationHints, usedHandSources);
    }
    for (const [key, object] of this.tiles) {
      if (!next.has(key)) {
        this.dynamic.remove(object.mesh);
        this.tiles.delete(key);
      }
    }
    for (const [key, context] of next) {
      this.applyTile(key, context, snapPersistentTiles);
    }
  }

  private reconcileWall(next: Map<string, SeatTileContext>): void {
    const count = Math.max(0, Math.min(108, this.snapshot?.wallRemaining ?? 0));
    if (count > this.wallRemaining || this.wallRemaining < 0) {
      this.wallVisible = new Set(WALL_DRAW_ORDER.slice(0, count));
    } else if (count < this.wallRemaining) {
      for (const index of WALL_DRAW_ORDER.slice(count, this.wallRemaining)) {
        this.wallVisible.delete(index);
      }
    }
    this.wallRemaining = count;
    // Blood Flow has 108 tiles: 54 two-tile stacks around the OpenRiichi
    // frame. The physical slots never move; draws only remove a slot from the
    // fixed engine draw order, so the remaining wall cannot jump between sides.
    for (const index of this.wallVisible) {
      next.set(`wall:${index}`, {
        seat: 4,
        tile: TILE_BACK_CELL,
        zone: "river",
        index,
        state: "normal",
      });
    }
  }

  private reconcileSeat(
    seat: number,
    next: Map<string, SeatTileContext>,
  ): void {
    const snapshot = this.snapshot!;
    const locked = expandHistogram(snapshot.lockedTiles[seat]!);
    const unlocked = expandHistogram(snapshot.unlockedHand);
    const local = seat === 0;
    const root = local ? expandHistogram(snapshot.winBase) : [];
    const winning = local
      ? subtractHistogram(snapshot.lockedTiles[seat]!, snapshot.winBase)
      : locked;
    const latestWinTile = latestWinTileFor(snapshot, seat);
    const legalDiscards = legalDiscardTiles(snapshot);
    const staged = new Map<number, number>();
    for (const tile of expandHistogram(snapshot.exchangeSelection)) {
      staged.set(tile, (staged.get(tile) ?? 0) + 1);
    }
    const pendingKeys = new Set(this.pendingExchangeSelectionKeys);
    const exchangeFull = snapshot.exchangeSelectedCount + pendingKeys.size >= 3;
    const missing = snapshot.missingSuits[0] ?? -1;
    const viewerCanSelectHand =
      snapshot.decisionActor === 0 &&
      (snapshot.phase === Phase.Exchange || snapshot.phase === Phase.Turn);
    const viewerCanDiscard =
      snapshot.decisionActor === 0 && snapshot.phase === Phase.Turn;
    const drawnTile = local ? snapshot.drawTile : -1;
    let drawnIndex = -1;
    if (drawnTile >= 0) {
      const hand = expandHistogram(unlocked);
      for (let index = hand.length - 1; index >= 0; index -= 1) {
        if (hand[index] === drawnTile) {
          drawnIndex = index;
          break;
        }
      }
    }

    if (local) {
      unlocked.forEach((tile, index) => {
        const selectedCount = staged.get(tile) ?? 0;
        const key = `hand:${seat}:${index}`;
        const selected =
          snapshot.phase === 0 && (pendingKeys.has(key) || selectedCount > 0);
        if (selected && !pendingKeys.has(key)) {
          staged.set(tile, selectedCount - 1);
        }
        let state: TileState = selected ? "selected" : "normal";
        if (snapshot.phase === 0 && exchangeFull && !selected) {
          state = "disabled";
        }
        const action = Action.DiscardOffset + tile;
        if (missing >= 0 && tileSuit(tile) === missing) state = "missing";
        if (viewerCanDiscard && !legalDiscards.has(tile)) {
          state = "disabled";
        }
        if (viewerCanDiscard && this.hintAction === action) state = "hint";
        next.set(key, {
          seat,
          tile,
          zone: "hand",
          index,
          layoutIndex:
            drawnIndex >= 0 && index > drawnIndex ? index - 1 : index,
          state,
          action: viewerCanSelectHand ? action : undefined,
          drawn: index === drawnIndex,
        });
      });
    } else {
      const count = snapshot.unlockedHandCounts[seat] ?? 0;
      for (let index = 0; index < count; index += 1) {
        next.set(`hand:${seat}:${index}`, {
          seat,
          tile: TILE_BACK_CELL,
          zone: "hand",
          index,
          state: "normal",
        });
      }
    }

    if (local) {
      // Keep the stable winning base in the hand row. It remains face-up but
      // grey and has no action id, so it cannot be selected or discarded.
      root.forEach((tile, index) => {
        next.set(`root:${seat}:${index}`, {
          seat,
          tile,
          zone: "root",
          index,
          state: "locked",
        });
      });

      // Only the public winning references leave the stable base. The newest
      // reference receives the bright treatment; historical references become
      // inert grey tiles as soon as the next win is recorded.
      const occurrences = new Map<number, number>();
      const totals = new Map<number, number>();
      const layout = localWinningReferenceLayout(snapshot, winning);
      for (const tile of winning) {
        totals.set(tile, (totals.get(tile) ?? 0) + 1);
      }
      winning.forEach((tile, index) => {
        const occurrence = occurrences.get(tile) ?? 0;
        occurrences.set(tile, occurrence + 1);
        const isLatest =
          latestWinTile === tile && occurrence === (totals.get(tile) ?? 1) - 1;
        const key = winReferenceKey(seat, tile, occurrence);
        next.set(key, {
          seat,
          tile,
          zone: "win",
          index: layout.get(key) ?? index,
          state: isLatest ? "winning" : "locked",
        });
      });
    } else {
      // Opponent bases are hidden by the engine. Their public winning
      // references still use a separate raised row.
      const latestIndex =
        latestWinTile >= 0 ? locked.lastIndexOf(latestWinTile) : -1;
      locked.forEach((tile, index) => {
        const isLatest = index === latestIndex;
        next.set(`locked:${seat}:${index}`, {
          seat,
          tile,
          zone: "locked",
          index,
          state: isLatest ? "winning" : "locked",
        });
      });
    }

    let meldIndex = 0;
    let meldOffset = MELD_START_X;
    for (const meld of snapshot.melds[seat] ?? []) {
      const copies = meld.kind === MeldKind.Pong ? 3 : 4;
      const width = meldWidth(meld.kind);
      for (let copy = 0; copy < copies; copy += 1) {
        const concealed =
          meld.kind === MeldKind.ConcealedKong && (copy === 1 || copy === 2);
        next.set(`meld:${seat}:${meldIndex}:${copy}`, {
          seat,
          tile: concealed ? TILE_BACK_CELL : meld.tile,
          zone: "meld",
          index: meldIndex * 4 + copy,
          state: "normal",
          meldKind: meld.kind,
          meldCopy: copy,
          meldOffset,
          sourceRelative: meld.sourceRelative,
        });
      }
      meldIndex += 1;
      meldOffset += width + TILE.x * 0.35;
    }

    let riverIndex = 0;
    snapshot.river.forEach((entry, globalIndex) => {
      if (entry.ownerRelative !== seat) return;
      next.set(`river:${seat}:${riverIndex}`, {
        seat,
        tile: entry.tile,
        zone: "river",
        index: riverIndex,
        state: globalIndex === snapshot.river.length - 1 ? "latest" : "normal",
      });
      riverIndex += 1;
    });
  }

  private applyTile(
    key: string,
    context: SeatTileContext,
    snapPersistentTile = false,
  ): void {
    const existing = this.tiles.get(key);
    const object =
      existing ?? this.createTile(key, context.tile, context.state);
    if (existing != null && object.tile !== context.tile) {
      object.mesh.geometry =
        this.geometries.get(context.tile) ??
        this.geometries.get(TILE_BACK_CELL)!;
    }
    const pose = this.poseFor(context);
    if (context.state === "latest") {
      pose.position.y += TILE.y * 0.28;
    }
    const snapToPose =
      snapPersistentTile ||
      existing == null ||
      context.zone === "root" ||
      context.zone === "win" ||
      context.zone === "meld" ||
      context.zone === "river";
    if (snapToPose) {
      object.mesh.position.copy(pose.position);
      object.mesh.rotation.set(
        pose.rotation.x,
        pose.rotation.y,
        pose.rotation.z,
      );
    }
    object.target.copy(pose.position);
    object.targetRotation.copy(pose.rotation);
    if (object.state !== context.state && this.atlas != null) {
      object.mesh.material = this.materialFor(
        context.state,
        this.atlas,
      ).clone();
    }
    object.state = context.state;
    object.tile = context.tile;
    const interactive =
      context.action != null &&
      context.seat === 0 &&
      context.state !== "disabled";
    object.mesh.userData = {
      key,
      tile: context.tile,
      action: context.action,
      interactive,
    };
    if (!interactive && this.hoverKey === key) {
      this.hoverKey = null;
      this.restoreStateTint(object);
    }
    object.mesh.visible =
      !this.hasTransientFor(key) && !this.hasHiddenTransientSource(key);
    if (existing == null) this.tiles.set(key, object);
  }

  private createDiscardAnimations(
    previous: UiSnapshot,
    next: Map<string, SeatTileContext>,
    animationHints: readonly AnimationHint[],
    usedHandSources: Set<string>,
  ): void {
    const previousRiverCounts = [0, 0, 0, 0];
    for (const entry of previous.river) {
      if (entry.ownerRelative >= 0 && entry.ownerRelative < 4) {
        previousRiverCounts[entry.ownerRelative]! += 1;
      }
    }
    const occurrences = [0, 0, 0, 0];
    for (const hint of animationHints) {
      if (hint.kind !== "discard") continue;
      const seat = hint.seatRelative;
      if (seat < 0 || seat >= 4) continue;
      const riverIndex = previousRiverCounts[seat]! + occurrences[seat]!;
      occurrences[seat]! += 1;
      const riverKey = `river:${seat}:${riverIndex}`;
      const context = next.get(riverKey);
      if (context == null) continue;
      const source = this.findDiscardSource(seat, hint.tile, usedHandSources);
      if (source == null) continue;
      usedHandSources.add(source.key);
      const pose = this.poseFor(context);
      if (context.state === "latest") {
        pose.position.y += TILE.y * 0.28;
      }
      this.addTransientTile(source.object, source.key, riverKey, pose);
    }
  }

  private createWinAnimations(
    previous: UiSnapshot,
    next: Map<string, SeatTileContext>,
    animationHints: readonly AnimationHint[],
    usedHandSources: Set<string>,
  ): void {
    const addedOccurrences = new Map<string, number>();
    for (const hint of animationHints) {
      if (hint.kind !== "hu") continue;
      const seat = hint.seatRelative;
      if (seat < 0 || seat >= 4) continue;

      const previousWinning = winningReferences(previous, seat);
      const occurrenceKey = `${seat}:${hint.tile}`;
      const occurrence =
        countTile(previousWinning, hint.tile) +
        (addedOccurrences.get(occurrenceKey) ?? 0);
      addedOccurrences.set(
        occurrenceKey,
        (addedOccurrences.get(occurrenceKey) ?? 0) + 1,
      );
      const targetKey = this.findWinTargetKey(
        next,
        seat,
        hint.tile,
        occurrence,
      );
      if (targetKey == null) continue;
      const context = next.get(targetKey);
      if (context == null || context.state !== "winning") continue;
      const source = this.findWinSource(
        seat,
        hint.sourceRelative,
        hint.tile,
        usedHandSources,
      );
      if (source == null) continue;
      if (source.key.startsWith("hand:")) usedHandSources.add(source.key);
      const pose = this.poseFor(context);
      this.addWinTransient(source.object, context.tile, targetKey, pose);
    }
  }

  private findWinTargetKey(
    next: ReadonlyMap<string, SeatTileContext>,
    seat: number,
    tile: number,
    occurrence: number,
  ): string | undefined {
    if (seat === 0) return `win:${seat}:${tile}:${occurrence}`;

    let currentOccurrence = 0;
    for (const [key, context] of next) {
      if (
        context.zone !== "locked" ||
        context.seat !== seat ||
        context.tile !== tile
      ) {
        continue;
      }
      if (currentOccurrence === occurrence) return key;
      currentOccurrence += 1;
    }
    return undefined;
  }

  private findDiscardSource(
    seat: number,
    tile: number,
    usedSources: ReadonlySet<string>,
  ): { key: string; object: TileObject } | undefined {
    const prefix = `hand:${seat}:`;
    const candidates: { key: string; object: TileObject }[] = [];
    for (const [key, object] of this.tiles) {
      if (!key.startsWith(prefix) || usedSources.has(key)) continue;
      if (seat === 0 && object.tile !== tile) continue;
      candidates.push({ key, object });
    }
    candidates.sort(
      (left, right) => handIndex(left.key) - handIndex(right.key),
    );
    return candidates.at(-1);
  }

  private findWinSource(
    winnerSeat: number,
    sourceSeat: number,
    tile: number,
    usedHandSources: ReadonlySet<string>,
  ): { key: string; object: TileObject } | undefined {
    if (sourceSeat >= 0 && sourceSeat < 4) {
      const riverCandidates = this.tileCandidates(
        `river:${sourceSeat}:`,
        (object) => object.tile === tile,
      );
      const riverSource = riverCandidates.at(-1);
      if (riverSource != null) return riverSource;

      // A robbed added kong has no river entry. Its visible source is the
      // matching public meld owned by the payer.
      const meldCandidates = this.tileCandidates(
        `meld:${sourceSeat}:`,
        (object) => object.tile === tile,
      );
      return meldCandidates.at(-1);
    }

    const handCandidates: { key: string; object: TileObject }[] = [];
    for (const [key, object] of this.tiles) {
      if (!key.startsWith(`hand:${winnerSeat}:`) || usedHandSources.has(key)) {
        continue;
      }
      if (winnerSeat === 0 && object.tile !== tile) continue;
      handCandidates.push({ key, object });
    }
    handCandidates.sort(
      (left, right) => handIndex(left.key) - handIndex(right.key),
    );
    return handCandidates.at(-1);
  }

  private tileCandidates(
    prefix: string,
    accepts: (object: TileObject) => boolean,
  ): { key: string; object: TileObject }[] {
    const candidates: { key: string; object: TileObject }[] = [];
    for (const [key, object] of this.tiles) {
      if (key.startsWith(prefix) && accepts(object)) {
        candidates.push({ key, object });
      }
    }
    candidates.sort(
      (left, right) => handIndex(left.key) - handIndex(right.key),
    );
    return candidates;
  }

  private addTransientTile(
    source: TileObject,
    sourceKey: string,
    riverKey: string,
    pose: { position: Vector3; rotation: Vector3 },
  ): void {
    const key = `discard:${this.transientSequence++}`;
    const material = (source.mesh.material as MeshPhongMaterial).clone();
    const mesh = new Mesh(source.mesh.geometry, material);
    mesh.position.copy(source.mesh.position);
    mesh.rotation.copy(source.mesh.rotation);
    mesh.scale.copy(source.mesh.scale);
    mesh.castShadow = source.mesh.castShadow;
    mesh.receiveShadow = source.mesh.receiveShadow;
    mesh.userData = { key, interactive: false };
    this.dynamic.add(mesh);
    source.mesh.visible = false;
    this.transientTiles.set(key, {
      mesh,
      sourceKey,
      hideSource: true,
      from: source.mesh.position.clone(),
      to: pose.position.clone(),
      fromRotation: new Vector3(
        source.mesh.rotation.x,
        source.mesh.rotation.y,
        source.mesh.rotation.z,
      ),
      toRotation: pose.rotation.clone(),
      startedAt: performance.now(),
      duration: 220,
      targetKey: riverKey,
    });
  }

  private addWinTransient(
    source: TileObject,
    tile: number,
    targetKey: string,
    pose: { position: Vector3; rotation: Vector3 },
  ): void {
    const key = `win:${this.transientSequence++}`;
    const geometry =
      this.geometries.get(tile) ?? this.geometries.get(TILE_BACK_CELL)!;
    const material = this.materialFor(
      "winning",
      this.atlas ?? new Texture(),
    ).clone();
    const mesh = new Mesh(geometry, material);
    mesh.position.copy(source.mesh.position);
    mesh.rotation.copy(source.mesh.rotation);
    mesh.scale.copy(source.mesh.scale);
    mesh.castShadow = source.mesh.castShadow;
    mesh.receiveShadow = source.mesh.receiveShadow;
    mesh.userData = { key, interactive: false };
    this.dynamic.add(mesh);
    this.transientTiles.set(key, {
      mesh,
      sourceKey: source.mesh.userData.key as string | undefined,
      hideSource: false,
      from: source.mesh.position.clone(),
      to: pose.position.clone(),
      fromRotation: new Vector3(
        source.mesh.rotation.x,
        source.mesh.rotation.y,
        source.mesh.rotation.z,
      ),
      toRotation: pose.rotation.clone(),
      startedAt: performance.now(),
      duration: 460,
      targetKey,
    });
  }

  private hasTransientFor(targetKey: string): boolean {
    for (const transient of this.transientTiles.values()) {
      if (transient.targetKey === targetKey) return true;
    }
    return false;
  }

  private hasHiddenTransientSource(sourceKey: string): boolean {
    for (const transient of this.transientTiles.values()) {
      if (transient.hideSource && transient.sourceKey === sourceKey)
        return true;
    }
    return false;
  }

  private clearTransientTiles(): void {
    for (const transient of this.transientTiles.values()) {
      this.dynamic.remove(transient.mesh);
      (transient.mesh.material as MeshPhongMaterial).dispose();
      if (transient.hideSource && transient.sourceKey != null) {
        const source = this.tiles.get(transient.sourceKey);
        if (source != null) source.mesh.visible = true;
      }
    }
    this.transientTiles.clear();
  }

  private createTile(key: string, tile: number, state: TileState): TileObject {
    const geometry =
      this.geometries.get(tile) ?? this.geometries.get(TILE_BACK_CELL)!;
    const mesh = new Mesh(
      geometry,
      this.materialFor(state, this.atlas ?? new Texture()).clone(),
    );
    mesh.scale.setScalar(1.5);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.dynamic.add(mesh);
    const object: TileObject = {
      mesh,
      target: new Vector3(),
      targetRotation: new Vector3(),
      state,
      tile,
    };
    this.tiles.set(key, object);
    void this.atlasPromise.then((atlas) => {
      this.atlas = atlas;
      const current = this.tiles.get(key);
      if (current != null)
        current.mesh.material = this.materialFor(state, atlas).clone();
    });
    return object;
  }

  private poseFor(context: SeatTileContext): {
    position: Vector3;
    rotation: Vector3;
  } {
    if (context.zone === "river") {
      if (context.seat === 4) {
        const slot = WALL_SLOTS[context.index] ?? {
          side: 0,
          stack: 0,
          layer: 0,
        };
        const local = new Vector3(
          8 * TILE.x - slot.stack * TILE.x,
          TILE.y / 2 + slot.layer * TILE.y,
          10 * TILE.x,
        );
        local.applyAxisAngle(new Vector3(0, 1, 0), (slot.side * Math.PI) / 2);
        return {
          position: local,
          rotation: new Vector3(Math.PI / 2, (slot.side * Math.PI) / 2, 0),
        };
      }
      const row = Math.floor(context.index / 6);
      const column = context.index % 6;
      const local = new Vector3(
        (column - 2.5) * TILE.x,
        TILE.y / 2,
        3 * TILE.x + TILE.z / 2 + row * TILE.z,
      );
      return {
        position: rotateSeat(local, context.seat),
        rotation: new Vector3(0, (context.seat * Math.PI) / 2, 0),
      };
    }

    if (context.zone === "root") {
      const count = expandHistogram(this.snapshot!.winBase).length;
      const local = new Vector3(
        (context.index - (count - 1) / 2) * TILE.x,
        TILE.y / 2,
        HAND_Z,
      );
      return {
        position: rotateSeat(local, context.seat),
        rotation: new Vector3(0.06 * Math.PI, (context.seat * Math.PI) / 2, 0),
      };
    }

    if (context.zone === "win") {
      const local = new Vector3(
        MELD_START_X + (context.index + 0.5) * TILE.x,
        TILE.y / 2 + TILE.y * 0.38,
        WIN_Z,
      );
      // Keep winning references above the left side of the local hand. The
      // drawn-tile slot stays clear on the right as later turns continue.
      return {
        position: rotateSeat(local, context.seat),
        rotation: new Vector3(
          context.seat === 0 ? 0.06 * Math.PI : 0,
          (context.seat * Math.PI) / 2,
          0,
        ),
      };
    }

    if (context.zone === "locked") {
      const count = expandHistogram(
        this.snapshot!.lockedTiles[context.seat]!,
      ).length;
      const local = new Vector3(
        (context.index - (count - 1) / 2) * TILE.x,
        TILE.y / 2 + TILE.y * 0.35,
        LOCKED_Z,
      );
      return {
        position: rotateSeat(local, context.seat),
        rotation: new Vector3(0, (context.seat * Math.PI) / 2, 0),
      };
    }

    if (context.zone === "meld") {
      const copy = context.meldCopy ?? context.index % 4;
      const kind = context.meldKind ?? MeldKind.ExposedKong;
      const sourceOffset =
        ((context.sourceRelative ?? context.seat) - context.seat + 4) % 4;
      const called = calledTileIndex(kind, sourceOffset);
      const start = context.meldOffset ?? MELD_START_X;
      if (kind === MeldKind.ConcealedKong) {
        const position = new Vector3(
          start + (copy + 0.5) * TILE.x,
          TILE.y / 2,
          MELD_Z,
        );
        return {
          position: rotateSeat(position, context.seat),
          rotation: new Vector3(0, (context.seat * Math.PI) / 2, 0),
        };
      }

      let cursor = start;
      const position = new Vector3();
      let calledYaw = 0;
      const addedTile =
        kind === MeldKind.AddedKong && called >= 0 ? called + 1 : -1;
      const handSideEdge = MELD_Z + TILE.z / 2;
      for (let current = 0; current <= copy; current += 1) {
        if (current === called || current === addedTile) {
          const depth = current === addedTile ? 1.5 * TILE.x : TILE.x / 2;
          position.set(cursor + TILE.z / 2, TILE.y / 2, handSideEdge - depth);
          calledYaw = Math.PI / 2;
          if (kind !== MeldKind.AddedKong || current === addedTile) {
            cursor += TILE.z;
          }
        } else {
          position.set(cursor + TILE.x / 2, TILE.y / 2, MELD_Z);
          cursor += TILE.x;
          calledYaw = 0;
        }
      }
      return {
        position: rotateSeat(position, context.seat),
        rotation: new Vector3(0, (context.seat * Math.PI) / 2 + calledYaw, 0),
      };
    }

    const localCount =
      context.seat === 0
        ? expandHistogram(this.snapshot!.unlockedHand).length
        : (this.snapshot!.unlockedHandCounts[context.seat] ?? 0);
    const activeCount =
      localCount - (context.seat === 0 && this.snapshot!.drawTile >= 0 ? 1 : 0);
    const layoutIndex = context.layoutIndex ?? context.index;
    const rootCount =
      context.seat === 0 ? expandHistogram(this.snapshot!.winBase).length : 0;
    const hasRoot = rootCount > 0;
    const activeStart = hasRoot
      ? ((rootCount - 1) / 2) * TILE.x + WIN_GAP + DRAWN_GAP
      : 0;
    const local = new Vector3(
      context.drawn
        ? hasRoot
          ? activeStart + activeCount * TILE.x + DRAWN_GAP
          : ((activeCount + 1) / 2) * TILE.x + DRAWN_GAP
        : hasRoot
          ? activeStart + layoutIndex * TILE.x
          : (layoutIndex - (activeCount - 1) / 2) * TILE.x,
      TILE.y / 2,
      HAND_Z,
    );
    // OpenRiichi rotates each player parent around Y and tilts only the
    // observed hand around X. Do not encode these independent axes as one
    // yaw value: doing so turns opponent tiles onto their sides.
    const pitch = context.seat === 0 ? 0.06 * Math.PI : 0;
    return {
      position: rotateSeat(local, context.seat),
      rotation: new Vector3(pitch, (context.seat * Math.PI) / 2, 0),
    };
  }

  private addLights(): void {
    this.scene.add(new AmbientLight(0xffffff, 0.2));
    const positions: [number, number, number, number][] = [
      [0, 20, 30, 18],
      [30, 10, 0, 12],
      [-30, 10, 0, 12],
      [0, 8, 0, 1],
    ];
    for (const [x, y, z, intensity] of positions) {
      const light = new PointLight(0xffffff, intensity, 0, 1.2);
      light.position.set(x, y, z);
      light.castShadow = true;
      light.shadow.mapSize.set(1024, 1024);
      this.scene.add(light);
    }
  }

  private resize = (): void => {
    const width = Math.max(1, this.canvas.clientWidth || window.innerWidth);
    const height = Math.max(1, this.canvas.clientHeight || window.innerHeight);
    const aspect = width / height;
    const verticalFov =
      aspect >= 1
        ? (2 *
            Math.atan(Math.tan((HORIZONTAL_FOV * Math.PI) / 360) / aspect) *
            180) /
          Math.PI
        : HORIZONTAL_FOV;
    this.camera.fov = verticalFov;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  };

  private render = (time: number): void => {
    if (this.disposed) return;
    const rawProgress = Math.min(
      1,
      (time - this.introStarted) / (INTRO_SECONDS * 1000),
    );
    const progress = this.reducedMotion
      ? 1
      : rawProgress * rawProgress * (3 - 2 * rawProgress);
    this.camera.position.lerpVectors(
      this.cameraStart,
      this.cameraEnd,
      progress,
    );
    this.currentLookTarget.lerpVectors(
      new Vector3(0, 2, 0),
      this.lookTarget,
      progress,
    );
    this.camera.lookAt(this.currentLookTarget);
    for (const object of this.tiles.values()) {
      const activeDuration = Math.max(80, this.animationDuration);
      const factor = this.reducedMotion
        ? 1
        : time < this.animationEnd
          ? 1 - Math.exp((-16.67 * 3) / activeDuration)
          : 0.22;
      object.mesh.position.lerp(object.target, factor);
      object.mesh.rotation.x +=
        (object.targetRotation.x - object.mesh.rotation.x) * factor;
      object.mesh.rotation.y +=
        (object.targetRotation.y - object.mesh.rotation.y) * factor;
      object.mesh.rotation.z +=
        (object.targetRotation.z - object.mesh.rotation.z) * factor;
    }
    for (const [key, transient] of this.transientTiles) {
      const progress = this.reducedMotion
        ? 1
        : Math.min(
            1,
            Math.max(0, (time - transient.startedAt) / transient.duration),
          );
      const eased = progress * progress * (3 - 2 * progress);
      transient.mesh.position.lerpVectors(transient.from, transient.to, eased);
      const rotation = new Vector3().lerpVectors(
        transient.fromRotation,
        transient.toRotation,
        eased,
      );
      transient.mesh.rotation.set(rotation.x, rotation.y, rotation.z);
      if (progress >= 1) {
        transient.mesh.position.copy(transient.to);
        transient.mesh.rotation.set(
          transient.toRotation.x,
          transient.toRotation.y,
          transient.toRotation.z,
        );
        this.dynamic.remove(transient.mesh);
        (transient.mesh.material as MeshPhongMaterial).dispose();
        this.transientTiles.delete(key);
        const target = this.tiles.get(transient.targetKey);
        if (target != null && !this.hasTransientFor(transient.targetKey)) {
          target.mesh.visible = true;
        }
        if (transient.hideSource && transient.sourceKey != null) {
          const source = this.tiles.get(transient.sourceKey);
          if (
            source != null &&
            !this.hasHiddenTransientSource(transient.sourceKey)
          ) {
            source.mesh.visible = true;
          }
        }
      }
    }
    this.renderer.render(this.scene, this.camera);
    this.animationFrame = requestAnimationFrame(this.render);
  };

  private bindInput(): void {
    this.canvas.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    this.canvas.addEventListener("pointerup", this.onPointerUp);
    this.canvas.addEventListener("pointerleave", this.onPointerLeave);
  }

  private unbindInput(): void {
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
    this.canvas.removeEventListener("pointerleave", this.onPointerLeave);
  }

  private intersection(event: PointerEvent): TileObject | undefined {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.dynamic.children, false);
    const hit = hits.find(
      (entry) => (entry.object as Mesh).userData.interactive,
    );
    if (hit == null) return undefined;
    const key = (hit.object as Mesh).userData.key as string | undefined;
    return key == null ? undefined : this.tiles.get(key);
  }

  private onPointerMove = (event: PointerEvent): void => {
    const object = this.intersection(event);
    const key = object == null ? null : (object.mesh.userData.key as string);
    if (key !== this.hoverKey) {
      if (this.hoverKey != null) this.setHover(this.hoverKey, false);
      this.hoverKey = key;
      if (key != null) this.setHover(key, true);
    }
  };

  private onPointerDown = (event: PointerEvent): void => {
    this.pressedKey =
      (this.intersection(event)?.mesh.userData.key as string | undefined) ??
      null;
  };

  private onPointerUp = (event: PointerEvent): void => {
    const object = this.intersection(event);
    const key = object?.mesh.userData.key as string | undefined;
    if (object != null && key != null && key === this.pressedKey) {
      const tile = object.mesh.userData.tile as number;
      if (tile >= 0 && tile < TILE_KIND_COUNT) {
        this.options.onTileClick(tile, key);
      }
    }
    this.pressedKey = null;
  };

  private onPointerLeave = (): void => {
    if (this.hoverKey != null) this.setHover(this.hoverKey, false);
    this.hoverKey = null;
    this.pressedKey = null;
  };

  private setHover(key: string, hovered: boolean): void {
    const object = this.tiles.get(key);
    if (object == null || (hovered && !object.mesh.userData.interactive))
      return;
    const material = object.mesh.material as MeshPhongMaterial;
    if (hovered) {
      material.emissive = new Color("#777700");
      material.emissiveIntensity = 0.5;
      return;
    }
    this.restoreStateTint(object);
  }

  private restoreStateTint(object: TileObject): void {
    const material = object.mesh.material as MeshPhongMaterial;
    // Restore the state tint after hover. A black reset would erase the gold
    // locked marker, missing-suit warning, or green NN hint.
    const base = this.materialFor(object.state, this.atlas ?? new Texture());
    material.emissive.copy(base.emissive);
    material.emissiveIntensity = base.emissiveIntensity;
  }
}

function rotateSeat(position: Vector3, seat: number): Vector3 {
  return position
    .clone()
    .applyAxisAngle(new Vector3(0, 1, 0), (seat * Math.PI) / 2);
}

function handIndex(key: string): number {
  const index = Number(key.slice(key.lastIndexOf(":") + 1));
  return Number.isFinite(index) ? index : -1;
}

function buildWallSlots(): WallSlot[] {
  const slots: WallSlot[] = [];
  for (let side = 0; side < WALL_SIDE_STACKS.length; side += 1) {
    for (let stack = 0; stack < WALL_SIDE_STACKS[side]!; stack += 1) {
      for (let layer = 0; layer < 2; layer += 1) {
        slots.push({ side, stack, layer });
      }
    }
  }
  return slots;
}

function buildWallDrawOrder(slots: readonly WallSlot[]): number[] {
  const bySide = [[], [], [], []] as number[][];
  slots.forEach((slot, index) => bySide[slot.side]!.push(index));
  const order: number[] = [];
  const maxLength = Math.max(...bySide.map((side) => side.length));
  for (let offset = 0; offset < maxLength; offset += 2) {
    for (const side of bySide) {
      if (side[offset] != null) order.push(side[offset]!);
      if (side[offset + 1] != null) order.push(side[offset + 1]!);
    }
  }
  return order;
}

function meldWidth(kind: number): number {
  if (kind === MeldKind.Pong) return 2 * TILE.x + TILE.z;
  if (kind === MeldKind.ConcealedKong) return 4 * TILE.x;
  return 3 * TILE.x + TILE.z;
}

function calledTileIndex(kind: number, sourceRelative: number): number {
  if (kind === MeldKind.Pong) {
    return sourceRelative === 1 ? 0 : sourceRelative === 2 ? 1 : 2;
  }
  if (kind === MeldKind.ExposedKong || kind === MeldKind.AddedKong) {
    return sourceRelative === 1 ? 0 : sourceRelative === 2 ? 1 : 3;
  }
  return -1;
}

function animationDurationForHint(hint: AnimationHint): number {
  switch (hint.kind) {
    case "draw":
    case "discard":
      return 150;
    case "meld":
      return 0;
    case "hu":
    case "payment":
      return 500;
    case "exchange_complete":
      return 200;
    case "missing_revealed":
      return 200;
    case "settlement_stage":
      return 500;
    case "game_end":
      return 500;
  }
}

function subtractHistogram(
  values: ArrayLike<number>,
  removed: ArrayLike<number>,
): number[] {
  const result = new Uint8Array(TILE_KIND_COUNT);
  for (let tile = 0; tile < TILE_KIND_COUNT; tile += 1) {
    result[tile] = Math.max(
      0,
      Number(values[tile] ?? 0) - Number(removed[tile] ?? 0),
    );
  }
  return expandHistogram(result);
}

function winningReferences(snapshot: UiSnapshot, seat: number): number[] {
  if (seat === 0) {
    return subtractHistogram(snapshot.lockedTiles[seat]!, snapshot.winBase);
  }
  return expandHistogram(snapshot.lockedTiles[seat]!);
}

function countTile(tiles: readonly number[], tile: number): number {
  let count = 0;
  for (const value of tiles) {
    if (value === tile) count += 1;
  }
  return count;
}

function winReferenceKey(
  seat: number,
  tile: number,
  occurrence: number,
): string {
  return `win:${seat}:${tile}:${occurrence}`;
}

function localWinningReferenceLayout(
  snapshot: UiSnapshot,
  winning: readonly number[],
): Map<string, number> {
  const available = new Set<string>();
  const availableOccurrences = new Map<number, number>();
  for (const tile of winning) {
    const occurrence = availableOccurrences.get(tile) ?? 0;
    availableOccurrences.set(tile, occurrence + 1);
    available.add(winReferenceKey(0, tile, occurrence));
  }

  const layout = new Map<string, number>();
  const historyOccurrences = new Map<number, number>();
  for (const event of snapshot.eventHistory) {
    if (event[0] !== EventKind.Hu || event[1] !== 0) continue;
    const tile = event[3];
    const occurrence = historyOccurrences.get(tile) ?? 0;
    historyOccurrences.set(tile, occurrence + 1);
    const key = winReferenceKey(0, tile, occurrence);
    if (available.has(key) && !layout.has(key)) {
      layout.set(key, layout.size);
    }
  }
  for (const key of available) {
    if (!layout.has(key)) layout.set(key, layout.size);
  }
  return layout;
}

function latestWinTileFor(snapshot: UiSnapshot, seat: number): number {
  for (let index = snapshot.stepEvents.length - 1; index >= 0; index -= 1) {
    const event = snapshot.stepEvents[index]!;
    if (event[0] === EventKind.Hu && event[1] === seat) return event[3];
  }
  for (let index = snapshot.eventHistory.length - 1; index >= 0; index -= 1) {
    const event = snapshot.eventHistory[index]!;
    if (event[0] === EventKind.Hu && event[1] === seat) return event[3];
  }
  return -1;
}
