#include "SpaceInvaders.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

SpaceInvaders::SpaceInvaders(std::uint32_t seed, int maxFrames)
    : mSeed(seed), mRandom(seed), mMaxFrames(maxFrames)
{
    if (maxFrames < 1) throw std::invalid_argument("maxFrames must be positive");
    restart();
}

std::uint32_t SpaceInvaders::random()
{
    mRandom ^= mRandom << 13;
    mRandom ^= mRandom >> 17;
    mRandom ^= mRandom << 5;
    return mRandom;
}

void SpaceInvaders::restart()
{
    mWorld = World{};
    mRandom = mSeed ^ 0x9e3779b9u;
    if (!mRandom) mRandom = 1;
    mWorld.invaders.fill((1u << Columns) - 1);
    for (int i = 0; i < ShieldCells; ++i)
        mWorld.shields[i] = (i % 18 >= 12 && (i % 6 == 2 || i % 6 == 3)) ? 0 : 2;
    mStateDirty = true;
}

void SpaceInvaders::reset(std::uint32_t seed)
{
    mSeed = seed;
    restart();
}

void SpaceInvaders::nextWave()
{
    ++mWorld.wave;
    mWorld.invaders.fill((1u << Columns) - 1);
    mWorld.remaining = EnemyCount;
    mWorld.formationX = 28;
    mWorld.formationY = 38 + std::min(mWorld.wave - 1, 5) * 4;
    mWorld.direction = 1;
    mWorld.moveTimer = 0;
    mWorld.fireTimer = 35;
    mWorld.bullets.fill(Bullet{});
    mWorld.playerShot.active = false;
}

void SpaceInvaders::moveFormation()
{
    const int interval = std::max(3, 26 - mWorld.wave * 2 - (EnemyCount - mWorld.remaining) / 3);
    if (++mWorld.moveTimer < interval) return;
    mWorld.moveTimer = 0;
    int left = Columns, right = -1, bottom = -1;
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < Columns; ++col)
            if (mWorld.invaders[row] & (1u << col))
            {
                left = std::min(left, col);
                right = std::max(right, col);
                bottom = row;
            }
    if (right < 0) return;
    const int step = 4 * mWorld.direction;
    if (enemyX(mWorld, left) + step < 8 || enemyX(mWorld, right) + step > Width - 8)
    {
        mWorld.direction = -mWorld.direction;
        mWorld.formationY += 8;
    }
    else mWorld.formationX += step;
    const int bottomY = enemyY(mWorld, bottom) + 5;
    if (bottomY >= 186) mWorld.shields.fill(0);
    if (bottomY >= PlayerY - 6) mWorld.terminal = true;
}

void SpaceInvaders::fireEnemyBullet()
{
    for (auto& bullet : mWorld.bullets)
    {
        if (bullet.active) continue;
        int columns[Columns], count = 0;
        for (int col = 0; col < Columns; ++col)
            for (int row = Rows - 1; row >= 0; --row)
                if (mWorld.invaders[row] & (1u << col))
                {
                    columns[count++] = col;
                    break;
                }
        if (!count) return;
        const int col = columns[random() % count];
        for (int row = Rows - 1; row >= 0; --row)
            if (mWorld.invaders[row] & (1u << col))
            {
                bullet.x = enemyX(mWorld, col);
                bullet.y = enemyY(mWorld, row) + 6;
                bullet.active = true;
                return;
            }
    }
}

bool SpaceInvaders::hitShield(int x, int oldY, int newY)
{
    if (std::max(oldY, newY) < 186 || std::min(oldY, newY) >= 198) return false;
    const int bunker = (x - 30) / 56;
    if (x < 30 || bunker >= 4) return false;
    const int column = (x - (30 + bunker * 56)) / 4;
    if (column >= 6) return false;
    for (int n = 0; n < 3; ++n)
    {
        const int row = newY < oldY ? 2 - n : n;
        const int cell = bunker * 18 + row * 6 + column;
        const int y = shieldY(cell);
        if (mWorld.shields[cell] && std::max(oldY, newY) >= y && std::min(oldY, newY) < y + 4)
        {
            --mWorld.shields[cell];
            return true;
        }
    }
    return false;
}

void SpaceInvaders::movePlayerShot()
{
    auto& shot = mWorld.playerShot;
    if (!shot.active) return;
    const int oldY = shot.y;
    shot.y -= 6;
    if (hitShield(shot.x, oldY, shot.y)) { shot.active = false; return; }
    // Swept vertical collision prevents fast bullets passing through targets.
    for (int row = Rows - 1; row >= 0; --row)
    {
        const int y = enemyY(mWorld, row);
        if (oldY < y - 5 || shot.y > y + 5) continue;
        for (int col = 0; col < Columns; ++col)
            if ((mWorld.invaders[row] & (1u << col)) && std::abs(shot.x - enemyX(mWorld, col)) <= 6)
            {
                mWorld.invaders[row] &= ~(1u << col);
                --mWorld.remaining;
                mWorld.score += row == 0 ? 30 : row < 3 ? 20 : 10;
                shot.active = false;
                return;
            }
    }
    if (mWorld.saucerActive && oldY >= 13 && shot.y <= 23 && std::abs(shot.x - mWorld.saucerX) <= 9)
    {
        mWorld.saucerActive = false;
        mWorld.score += 100;
        shot.active = false;
    }
    if (shot.y < 0) shot.active = false;
}

void SpaceInvaders::moveEnemyBullets()
{
    for (auto& bullet : mWorld.bullets)
    {
        if (!bullet.active) continue;
        const int oldY = bullet.y;
        bullet.y += 3 + std::min(mWorld.wave / 3, 2);
        if (hitShield(bullet.x, oldY, bullet.y)) { bullet.active = false; continue; }
        if (!mWorld.invulnerable && oldY <= PlayerY + 4 && bullet.y >= PlayerY - 4 &&
            std::abs(bullet.x - mWorld.playerX) <= 7)
        {
            --mWorld.lives;
            mWorld.invulnerable = 90;
            mWorld.playerX = Width / 2;
            mWorld.playerShot.active = false;
            mWorld.bullets.fill(Bullet{});
            if (!mWorld.lives) mWorld.terminal = true;
            break;
        }
        if (bullet.y >= Height) bullet.active = false;
    }
}

int SpaceInvaders::do_action(int action)
{
    if (mWorld.terminal) return 0;
    const int score = mWorld.score;
    action = (action % ActionCount + ActionCount) % ActionCount;
    ++mWorld.frame;
    if (mWorld.invulnerable) --mWorld.invulnerable;
    if (mWorld.cooldown) --mWorld.cooldown;
    if (action == Left || action == LeftFire) mWorld.playerX = std::max(8, mWorld.playerX - 3);
    if (action == Right || action == RightFire) mWorld.playerX = std::min(Width - 8, mWorld.playerX + 3);
    if ((action == Fire || action == LeftFire || action == RightFire) &&
        !mWorld.playerShot.active && !mWorld.cooldown)
    {
        mWorld.playerShot.x = mWorld.playerX;
        mWorld.playerShot.y = PlayerY - 7;
        mWorld.playerShot.active = true;
        mWorld.cooldown = 8;
    }
    moveFormation();
    movePlayerShot();
    moveEnemyBullets();
    if (--mWorld.fireTimer <= 0)
    {
        fireEnemyBullet();
        mWorld.fireTimer = std::max(10, 40 - mWorld.wave * 3) + random() % 12;
    }
    if (mWorld.saucerActive)
    {
        mWorld.saucerX += 2 * mWorld.saucerDirection;
        if (mWorld.saucerX < 0 || mWorld.saucerX >= Width) mWorld.saucerActive = false;
    }
    else if (--mWorld.saucerTimer <= 0)
    {
        mWorld.saucerActive = true;
        mWorld.saucerDirection = (random() & 1) ? 1 : -1;
        mWorld.saucerX = mWorld.saucerDirection > 0 ? 0 : Width - 1;
        mWorld.saucerTimer = 600;
    }
    if (!mWorld.remaining && !mWorld.terminal)
    {
        mWorld.score += 100;
        nextWave();
    }
    if (mWorld.frame >= mMaxFrames) mWorld.terminal = true;
    mStateDirty = true;
    return mWorld.score - score;
}

byte_t* SpaceInvaders::getState()
{
    if (!mStateDirty) return mState.data();
    mState.fill(0);
    mState[0] = mWorld.playerX;
    mState[1] = mWorld.lives;
    mState[2] = mWorld.playerShot.active;
    mState[3] = mWorld.playerShot.active ? mWorld.playerShot.x : 0;
    mState[4] = mWorld.playerShot.active ? mWorld.playerShot.y : 0;
    mState[5] = std::max(0, mWorld.formationX);
    mState[6] = mWorld.formationY;
    mState[7] = mWorld.direction > 0 ? 1 : 0;
    mState[8] = mWorld.remaining;
    mState[9] = std::min(255, mWorld.wave);
    mState[10] = mWorld.cooldown;
    mState[11] = mWorld.terminal;
    int nearest = -1;
    for (int i = 0; i < MaxBullets; ++i)
    {
        const auto& b = mWorld.bullets[i];
        mState[71 + 3 * i] = b.active;
        mState[72 + 3 * i] = b.active ? b.x : 0;
        mState[73 + 3 * i] = b.active ? b.y : 0;
        if (b.active && (nearest < 0 || b.y > mWorld.bullets[nearest].y)) nearest = i;
    }
    if (nearest >= 0)
    {
        mState[12] = mWorld.bullets[nearest].x;
        mState[13] = mWorld.bullets[nearest].y;
    }
    mState[14] = mWorld.score & 255;
    mState[15] = (mWorld.score >> 8) & 255;
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < Columns; ++col)
            mState[16 + row * Columns + col] = (mWorld.invaders[row] >> col) & 1;
    for (int i = 0; i < ShieldCells; ++i)
        mState[95 + i / 4] |= mWorld.shields[i] << ((i % 4) * 2);
    mState[113] = mWorld.frame & 255;
    mState[114] = (mWorld.frame >> 8) & 255;
    mState[115] = mWorld.saucerActive;
    mState[116] = mWorld.saucerActive ? mWorld.saucerX : 0;
    mState[117] = mWorld.saucerDirection > 0;
    mState[118] = mWorld.invulnerable;
    // Signed formation origin: byte 5 is the clamped coordinate; 119..120 encode origin + 256.
    mState[119] = (mWorld.formationX + 256) & 255;
    mState[120] = (mWorld.formationX + 256) >> 8;
    mStateDirty = false;
    return mState.data();
}

int SpaceInvaders::demoAction() const
{
    for (const auto& bullet : mWorld.bullets)
        if (bullet.active && bullet.y > 165 && std::abs(bullet.x - mWorld.playerX) < 13)
            return mWorld.playerX < 128 ? RightFire : LeftFire;
    int target = 128, distance = Width;
    for (int col = 0; col < Columns; ++col)
        for (int row = Rows - 1; row >= 0; --row)
            if (mWorld.invaders[row] & (1u << col))
            {
                const int x = enemyX(mWorld, col);
                if (std::abs(x - mWorld.playerX) < distance)
                {
                    target = x;
                    distance = std::abs(x - mWorld.playerX);
                }
                break;
            }
    if (target < mWorld.playerX - 3) return LeftFire;
    if (target > mWorld.playerX + 3) return RightFire;
    return Fire;
}
