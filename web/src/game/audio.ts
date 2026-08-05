export type SoundName =
  | "tile"
  | "draw"
  | "discard"
  | "slide"
  | "flip"
  | "reveal"
  | "click"
  | "mouse_over"
  | "score_count"
  | "hint"
  | "fade_in";

export class GameAudio {
  muted = false;

  private readonly sounds = new Map<SoundName, HTMLAudioElement>();
  private unlocked = false;

  unlock(): void {
    this.unlocked = true;
  }

  play(name: SoundName): void {
    if (this.muted || !this.unlocked) return;
    let sound = this.sounds.get(name);
    if (sound == null) {
      sound = new Audio(`${import.meta.env.BASE_URL}assets/audio/${name}.opus`);
      sound.preload = "auto";
      this.sounds.set(name, sound);
    }
    sound.currentTime = 0;
    void sound.play().catch(() => undefined);
  }
}
