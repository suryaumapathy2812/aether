import { directAgentFetch } from "@/lib/api";

export async function getUserPreference(userId: string, key: string): Promise<string | null> {
  try {
    const res = await directAgentFetch(
      `/agent/v1/preferences/get?user_id=${encodeURIComponent(userId)}&key=${encodeURIComponent(key)}`,
    );
    if (!res.ok) {
      if (res.status === 404) return null;
      throw new Error("Failed to get preference");
    }
    const data = await res.json();
    return data.value || null;
  } catch {
    return null;
  }
}

export async function setUserPreference(
  userId: string,
  key: string,
  value: string,
): Promise<boolean> {
  try {
    const res = await directAgentFetch("/agent/v1/preferences/set", {
      method: "POST",
      body: JSON.stringify({
        user_id: userId,
        key,
        value,
      }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function deleteUserPreference(userId: string, key: string): Promise<boolean> {
  try {
    const res = await directAgentFetch("/agent/v1/preferences/delete", {
      method: "POST",
      body: JSON.stringify({
        user_id: userId,
        key,
      }),
    });
    return res.ok;
  } catch {
    return false;
  }
}
