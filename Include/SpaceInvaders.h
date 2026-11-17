#pragma once

#include "Game.h"
#include <array>
#include <cstdint>

// Integer, fixed-step simulation. No allocation, rendering, or wall-clock work in step().
class SpaceInvaders final : public Game
{
public:
    enum { Width = 256, Height = 240, PlayerY = 222, Rows = 5, Columns = 11,
           EnemyCount = Rows * Columns, MaxBullets = 8, ShieldCells = 72, StateSize = 128 };
    enum Action { Noop, Fire, Left, Right, LeftFire, RightFire, ActionCount };
    struct Bullet { int x = 0, y = 0; bool active = false; };
    struct World
    {
        int playerX = 128, lives = 3, score = 0, frame = 0, wave = 1;
        int formationX = 28, formationY = 38, direction = 1, remaining = EnemyCount;
        int moveTimer = 0, fireTimer = 35, cooldown = 0, invulnerable = 0;
        int saucerX = 0, saucerDirection = 1, saucerTimer = 600;
        bool saucerActive = false, terminal = false;
        std::array<std::uint16_t, Rows> invaders{};
        Bullet playerShot;
        std::array<Bullet, MaxBullets> bullets{};
        std::array<byte_t, ShieldCells> shields{};
    };

    explicit SpaceInvaders(std::uint32_t seed = 123, int maxFrames = 3600);
    void restart() override;
    void reset(std::uint32_t seed);
    int do_action(int action) override;
    bool is_over() override { return mWorld.terminal; }
    byte_t* getState() override;
    int getStateSize() override { return StateSize; }
    int getActionCount() const override { return ActionCount; }
    const World& world() const { return mWorld; }
    static int enemyX(const World& w, int column) { return w.formationX + column * 18; }
    static int enemyY(const World& w, int row) { return w.formationY + row * 16; }
    static int shieldX(int cell) { return 30 + (cell / 18) * 56 + (cell % 6) * 4; }
    static int shieldY(int cell) { return 186 + ((cell % 18) / 6) * 4; }
    // A simple built-in pilot for demonstrations and repeatable benchmarks.
    int demoAction() const;

private:
    World mWorld;
    std::array<byte_t, StateSize> mState{};
    std::uint32_t mSeed, mRandom;
    int mMaxFrames;
    bool mStateDirty = true;
    std::uint32_t random();
    void nextWave();
    void moveFormation();
    void fireEnemyBullet();
    bool hitShield(int x, int oldY, int newY);
    void movePlayerShot();
    void moveEnemyBullets();
    friend struct SpaceInvadersTestAccess;
};
