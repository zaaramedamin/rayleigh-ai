import { afterEach, describe, expect, it, vi } from "vitest";
import { LocationError, pinpoint } from "./location";

afterEach(() => vi.unstubAllGlobals());

type Success = (p: { coords: { latitude: number; longitude: number; accuracy: number } }) => void;
type Failure = (e: { code: number; PERMISSION_DENIED: number; TIMEOUT: number }) => void;

function stubGeolocation(behaviour: (ok: Success, fail: Failure) => void) {
  vi.stubGlobal("navigator", { geolocation: { getCurrentPosition: behaviour } });
}

describe("pinpoint", () => {
  it("turns the browser's position into a place, with its accuracy", async () => {
    stubGeolocation((ok) => ok({ coords: { latitude: 36.8, longitude: 10.18, accuracy: 41.6 } }));

    const place = await pinpoint();

    expect(place).toMatchObject({ lat: 36.8, lon: 10.18, source: "gps", approximate: false, accuracyMeters: 42 });
  });

  it("explains a blocked permission", async () => {
    stubGeolocation((_ok, fail) => fail({ code: 1, PERMISSION_DENIED: 1, TIMEOUT: 3 }));

    await expect(pinpoint()).rejects.toThrow(/blocked/i);
  });

  it("explains a timeout and an unavailable position differently", async () => {
    stubGeolocation((_ok, fail) => fail({ code: 3, PERMISSION_DENIED: 1, TIMEOUT: 3 }));
    await expect(pinpoint()).rejects.toThrow(/too long/i);

    stubGeolocation((_ok, fail) => fail({ code: 2, PERMISSION_DENIED: 1, TIMEOUT: 3 }));
    await expect(pinpoint()).rejects.toThrow(/not available/i);
  });

  it("says so when the browser cannot locate at all", async () => {
    vi.stubGlobal("navigator", {});

    await expect(pinpoint()).rejects.toBeInstanceOf(LocationError);
  });
});
